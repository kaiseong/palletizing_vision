"""Deterministic pallet recording replay, metrics, JSONL, and MP4 overlay."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .calibration import factory_extrinsics_to_transform
from .output import to_jsonable
from .pallet_acquisition import (
    AcquisitionConfig,
    HoleGateStatus,
    LCornerGateStatus,
    StationaryHoleGate,
    StationaryLCornerGate,
)
from .pallet_geometry import PalletStackEstimator
from .pallet_models import (
    HeldBoxHint,
    NOMINAL_READY_T_BASE_FROM_HEAD,
    PalletEstimatorConfig,
    PalletSceneObservation,
    load_pallet_estimator_config,
    load_slot1_hole_reference,
)
from .pallet_visualization import draw_pallet_overlay
from .recording import SessionReader
from .session import SessionMetadata
from .transforms import transform_from_euler_zyx, validate_transform


FloatArray = NDArray[np.float64]


ACQUISITION_AUDIT_FIRST_FRAME = 72
ACQUISITION_AUDIT_LAST_FRAME = 105
ACQUISITION_AUDIT_WARMUP_LAST_FRAME = 75
ACQUISITION_AUDIT_FIRST_EVALUATED_FRAME = 76
ACQUISITION_AUDIT_REQUIRED_STABLE_FRAMES = 24
ACQUISITION_AUDIT_EXPECTED_EVALUATED_FRAMES = 30


def _cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to create a pallet replay MP4") from exc
    return cv2


def _root_payload(
    root_config: Mapping[str, Any] | str | Path | PalletEstimatorConfig,
) -> Mapping[str, Any]:
    if isinstance(root_config, PalletEstimatorConfig):
        return {}
    if isinstance(root_config, Mapping):
        return root_config
    path = Path(root_config)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read pallet configuration {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("pallet configuration root must be an object")
    return value


def _replay_base_from_depth(
    metadata: SessionMetadata,
    root: Mapping[str, Any],
) -> tuple[FloatArray, dict[str, Any]]:
    calibration = root.get("calibration", {})
    if not isinstance(calibration, Mapping):
        raise ValueError("calibration configuration must be an object")
    configured_head = calibration.get("T_base_from_head_ready_audit")
    T_base_head = (
        np.asarray(configured_head, dtype=np.float64)
        if configured_head is not None
        else np.asarray(NOMINAL_READY_T_BASE_FROM_HEAD, dtype=np.float64)
    )
    validate_transform(T_base_head, name="T_base_from_head_ready_audit")

    configured_mount = calibration.get("T_head_from_color")
    if configured_mount is not None:
        T_head_color = validate_transform(configured_mount, name="T_head_from_color")
        mount_source = "root_config_matrix"
    else:
        nominal = metadata.nominal_transform
        translation = nominal.get("translation_m")
        euler = nominal.get("euler_zyx_deg")
        if translation is None or euler is None:
            raise ValueError("recording metadata lacks nominal head/color transform")
        T_head_color = transform_from_euler_zyx(
            translation,
            float(euler[0]),
            float(euler[1]),
            float(euler[2]),
            degrees=True,
        )
        mount_source = "recording_nominal_transform"

    recorded_E = factory_extrinsics_to_transform(metadata.depth_to_color)
    configured_E = calibration.get("E_color_from_depth")
    configured_extrinsics_match = True
    if configured_E is not None:
        configured_extrinsics_match = bool(
            np.allclose(
                validate_transform(configured_E, name="configured_E_color_from_depth"),
                recorded_E,
                rtol=0.0,
                atol=1e-7,
            )
        )
    transform = T_base_head @ T_head_color @ recorded_E
    correction = np.asarray(
        calibration.get("base_translation_correction_m", (0.0, 0.0, 0.0)),
        dtype=np.float64,
    )
    if correction.shape != (3,) or not np.all(np.isfinite(correction)):
        raise ValueError(
            "base_translation_correction_m must be a finite length-3 vector"
        )
    transform = np.asarray(transform, dtype=np.float64).copy()
    transform[:3, 3] += correction
    validate_transform(transform, name="nominal_ready_T_base_depth")
    return transform, {
        "calibration_status": "nominal_ready_assumed",
        "T_base_from_head_source": (
            "root_config_ready_audit"
            if configured_head is not None
            else "compiled_ready_audit"
        ),
        "T_head_from_color_source": mount_source,
        "E_color_from_depth_source": "recorded_active_profile",
        "configured_extrinsics_match_recording": configured_extrinsics_match,
        "base_translation_correction_m": correction.tolist(),
        "base_translation_correction_applied_once": True,
        "absolute_base_validated": False,
    }


def _profile_diagnostics(
    metadata: SessionMetadata, root: Mapping[str, Any]
) -> dict[str, Any]:
    expected = root.get("camera", {})
    if not isinstance(expected, Mapping):
        expected = {}
    depth = metadata.depth_profile
    color = metadata.color_profile
    checks = {
        "camera_serial": not expected.get("serial")
        or str(expected["serial"]) == metadata.camera_serial,
        "depth_640x480_30": (
            depth.intrinsics.width == 640
            and depth.intrinsics.height == 480
            and depth.intrinsics.fps == 30
            and depth.format.lower() == "z16"
        ),
        "color_640x480_30": (
            color.intrinsics.width == 640
            and color.intrinsics.height == 480
            and color.intrinsics.fps == 30
        ),
        "depth_intrinsics_finite": bool(
            np.all(
                np.isfinite(
                    (
                        depth.intrinsics.fx,
                        depth.intrinsics.fy,
                        depth.intrinsics.cx,
                        depth.intrinsics.cy,
                    )
                )
            )
        ),
    }
    return {
        "checks": checks,
        "valid_for_commissioning_evidence": bool(all(checks.values())),
        "camera_serial": metadata.camera_serial,
        "depth_profile": depth.to_dict(),
        "color_profile": color.to_dict(),
    }


def _replay_held_hint(root: Mapping[str, Any]) -> HeldBoxHint | None:
    held = root.get("held_box", {})
    if not isinstance(held, Mapping):
        return None
    right_raw = held.get("nominal_ready_right_eef_base_xyz_m")
    left_raw = held.get("nominal_ready_left_eef_base_xyz_m")
    if right_raw is None or left_raw is None:
        return None
    right = np.asarray(right_raw, dtype=np.float64)
    left = np.asarray(left_raw, dtype=np.float64)
    offset = np.asarray(
        held.get("center_offset_from_eef_midpoint_m", (0.0, 0.0, 0.0)),
        dtype=np.float64,
    )
    if any(value.shape != (3,) for value in (right, left, offset)):
        raise ValueError("replay EEF origins and held-box offset must be XYZ vectors")
    if not np.all(np.isfinite([right, left, offset])):
        raise ValueError("replay EEF origins and held-box offset must be finite")
    separation = left - right
    if np.linalg.norm(separation[:2]) <= 1e-6:
        raise ValueError("replay EEF separation is degenerate")
    yaw = math.atan2(float(separation[1]), float(separation[0])) + math.radians(
        float(held.get("long_axis_from_eef_line_offset_deg", 0.0))
    )
    size = tuple(
        float(value) for value in held.get("nominal_size_m", (0.4, 0.253, 0.16))
    )
    return HeldBoxHint(
        center_base=0.5 * (right + left) + offset,
        yaw_base_rad=yaw,
        eef_proxy_z_base_m=float(0.5 * (right[2] + left[2])),
        footprint_size_m=(size[0], size[1]),
    )


def _finite_float(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _fixed_ready_clearance_audit(
    root: Mapping[str, Any],
    *,
    stack_upper_bound_m: float | None,
    maximum_box_height_m: float,
    minimum_clearance_m: float,
) -> dict[str, Any]:
    held = root.get("held_box", {})
    if not isinstance(held, Mapping):
        return {
            "applicable": False,
            "status": "unobservable_missing_held_box_config",
            "source": "configured_fixed_ready_dual_eef_nominal_geometry",
            "motion_authorized_from_this_audit": False,
        }
    required_keys = (
        "nominal_ready_right_eef_base_xyz_m",
        "nominal_ready_left_eef_base_xyz_m",
    )
    missing_keys = [key for key in required_keys if held.get(key) is None]
    if missing_keys:
        return {
            "applicable": False,
            "status": "unobservable_missing_fixed_ready_eef_config",
            "missing_keys": missing_keys,
            "source": "configured_fixed_ready_dual_eef_nominal_geometry",
            "motion_authorized_from_this_audit": False,
        }
    right = np.asarray(held["nominal_ready_right_eef_base_xyz_m"], dtype=np.float64)
    left = np.asarray(held["nominal_ready_left_eef_base_xyz_m"], dtype=np.float64)
    offset = np.asarray(
        held.get("center_offset_from_eef_midpoint_m", (0.0, 0.0, 0.0)),
        dtype=np.float64,
    )
    if any(value.shape != (3,) for value in (right, left, offset)):
        raise ValueError(
            "fixed-ready EEF origins and held-box offset must be XYZ vectors"
        )
    if not np.all(np.isfinite([right, left, offset])):
        raise ValueError("fixed-ready EEF origins and held-box offset must be finite")
    box_bottom_uncertainty_m = _finite_float(
        held.get("fixed_ready_box_bottom_uncertainty_m", 0.015),
        name="held_box.fixed_ready_box_bottom_uncertainty_m",
    )
    if box_bottom_uncertainty_m < 0.0:
        raise ValueError(
            "held_box.fixed_ready_box_bottom_uncertainty_m must be non-negative"
        )
    eef_midpoint = 0.5 * (right + left)
    center = eef_midpoint + offset
    nominal_box_bottom_z_m = float(center[2]) - 0.5 * maximum_box_height_m
    conservative_box_bottom_z_m = nominal_box_bottom_z_m - box_bottom_uncertainty_m
    clearance_lower_bound_m = (
        None
        if stack_upper_bound_m is None
        else conservative_box_bottom_z_m - stack_upper_bound_m
    )
    gate_passed = (
        None
        if clearance_lower_bound_m is None
        else bool(clearance_lower_bound_m >= minimum_clearance_m)
    )
    return {
        "applicable": True,
        "source": "configured_fixed_ready_dual_eef_nominal_geometry",
        "right_eef_base_xyz_m": right.tolist(),
        "left_eef_base_xyz_m": left.tolist(),
        "eef_midpoint_base_xyz_m": eef_midpoint.tolist(),
        "center_offset_from_eef_midpoint_m": offset.tolist(),
        "configured_center_base_xyz_m": center.tolist(),
        "maximum_box_height_m": maximum_box_height_m,
        "fixed_ready_box_bottom_uncertainty_m": box_bottom_uncertainty_m,
        "nominal_box_bottom_z_base_m": nominal_box_bottom_z_m,
        "conservative_box_bottom_z_base_m": conservative_box_bottom_z_m,
        "conservative_stack_top_z_base_m": stack_upper_bound_m,
        "required_lower_bound_m": minimum_clearance_m,
        "observed_conservative_lower_bound_m": clearance_lower_bound_m,
        "review_only_gate_passed": gate_passed,
        "motion_authorized_from_this_audit": False,
        "status": (
            "unobservable_no_valid_stack_top"
            if clearance_lower_bound_m is None
            else "pass_review_only_nominal_geometry"
            if clearance_lower_bound_m >= minimum_clearance_m
            else "fail_review_only_raise_ready_pose"
        ),
    }


def _angle_std_rad(values: Sequence[float]) -> float | None:
    if not values:
        return None
    unwrapped = np.unwrap(np.asarray(values, dtype=np.float64))
    return float(np.std(unwrapped))


def _longest_true_run(values: Sequence[bool]) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return longest


def _best_one_second_valid_ratio(
    timestamps_s: Sequence[float], valid: Sequence[bool]
) -> float:
    if not timestamps_s:
        return 0.0
    times = np.asarray(timestamps_s, dtype=np.float64)
    flags = np.asarray(valid, dtype=np.float64)
    best = 0.0
    right = 0
    for left in range(len(times)):
        right = max(right, left)
        while right + 1 < len(times) and times[right + 1] - times[left] <= 1.0:
            right += 1
        best = max(best, float(np.mean(flags[left : right + 1])))
    return best


def _has_stable_window(
    observations: Sequence[PalletSceneObservation],
    *,
    length: int,
    center_spread_m: float,
    yaw_spread_rad: float,
) -> bool:
    if length <= 0:
        return True
    for start in range(0, len(observations) - length + 1):
        window = observations[start : start + length]
        if not all(item.valid for item in window):
            continue
        centers = np.stack([item.stack.center_base for item in window])
        yaws = np.unwrap(np.asarray([item.stack.yaw_base_rad for item in window]))
        if (
            float(np.max(np.ptp(centers[:, :2], axis=0))) <= center_spread_m
            and float(np.ptp(yaws)) <= yaw_spread_rad
        ):
            return True
    return False


def _prepare_output(path: str | Path | None, *, overwrite: bool) -> Path | None:
    if path is None:
        return None
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _gate_payload(
    status: LCornerGateStatus | HoleGateStatus | None,
) -> dict[str, Any] | None:
    if status is None:
        return None
    return to_jsonable(asdict(status))


def _session_acceptance(
    session_name: str,
    *,
    processed_ratio: float,
    valid_ratio: float,
    rejected_without_reason: int,
    branch_flips: int,
    branch_image_right_ratio: float,
    center_std_xyz_m: Sequence[float] | None,
    yaw_std_rad: float | None,
    stable_window: bool,
    best_one_second_ratio: float,
    maximum_center_jump_m: float | None,
    maximum_opening_error_m: float | None,
    maximum_plane_p95_m: float | None,
    maximum_orthogonality_rad: float | None,
    config: PalletEstimatorConfig,
) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "decodable_frame_ratio_ge_0_90": processed_ratio >= 0.90,
        "rejected_frames_have_reason": rejected_without_reason == 0,
        "axis_branch_flip_count_zero": branch_flips == 0,
        "accepted_opening_error_within_gate": (
            maximum_opening_error_m is not None
            and maximum_opening_error_m <= config.gates.max_opening_size_error_m
        ),
        "accepted_plane_p95_within_gate": (
            maximum_plane_p95_m is not None
            and maximum_plane_p95_m <= config.gates.max_plane_p95_residual_m
        ),
        "accepted_orthogonality_within_gate": (
            maximum_orthogonality_rad is not None
            and maximum_orthogonality_rad <= config.gates.max_orthogonality_error_rad
        ),
    }
    if session_name == "pallet_1_arrived":
        checks.update(
            {
                "valid_ratio_ge_0_90": valid_ratio >= 0.90,
                "center_std_each_xy_le_0_005_m": (
                    center_std_xyz_m is not None and max(center_std_xyz_m[:2]) <= 0.005
                ),
                "yaw_std_le_1_5_deg": (
                    yaw_std_rad is not None and yaw_std_rad <= math.radians(1.5)
                ),
                "stable_five_frame_window": stable_window,
                "image_right_branch_ratio_1_0": branch_image_right_ratio == 1.0,
            }
        )
    elif session_name == "pallet_data":
        checks.update(
            {
                "valid_ratio_ge_0_50": valid_ratio >= 0.50,
                "best_one_second_valid_ratio_ge_0_80": best_one_second_ratio >= 0.80,
                "maximum_consecutive_center_jump_le_0_030_m": (
                    maximum_center_jump_m is not None and maximum_center_jump_m <= 0.030
                ),
            }
        )
    elif session_name == "pallet_box":
        checks.update(
            {
                "valid_ratio_ge_0_50": valid_ratio >= 0.50,
                "best_one_second_valid_ratio_ge_0_80": best_one_second_ratio >= 0.80,
                "all_accepted_openings_within_gate": (
                    maximum_opening_error_m is not None
                    and maximum_opening_error_m <= config.gates.max_opening_size_error_m
                ),
            }
        )
    return {"passed": bool(all(checks.values())), "checks": checks}


def evaluate_pallet_session(
    session_path: str | Path,
    root_config: Mapping[str, Any] | str | Path | PalletEstimatorConfig,
    *,
    output_mp4: str | Path | None = None,
    output_summary: str | Path | None = None,
    output_jsonl: str | Path | None = None,
    output_config_snapshot: str | Path | None = None,
    overwrite: bool = False,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Replay one RGB-D recording without connecting to the robot."""

    if max_frames is not None and int(max_frames) <= 0:
        raise ValueError("max_frames must be positive")
    mp4_path = _prepare_output(output_mp4, overwrite=overwrite)
    summary_path = _prepare_output(output_summary, overwrite=overwrite)
    jsonl_path = _prepare_output(output_jsonl, overwrite=overwrite)
    config_snapshot_path = _prepare_output(
        output_config_snapshot,
        overwrite=overwrite,
    )

    reader = SessionReader(session_path)
    session_name = Path(session_path).name
    root = _root_payload(root_config)
    estimator_config = load_pallet_estimator_config(root_config)
    pallet_block = root.get("pallet", {}) if root else {}
    slot1_hole_reference = (
        load_slot1_hole_reference(root)
        if isinstance(pallet_block, Mapping)
        and "slot1_hole_reference" in pallet_block
        else None
    )
    acquisition_config = AcquisitionConfig.from_root_config(root)
    T_base_depth, registration = _replay_base_from_depth(reader.metadata, root)
    profile = _profile_diagnostics(reader.metadata, root)
    estimator = PalletStackEstimator(estimator_config)
    l_corner_gate = StationaryLCornerGate(
        acquisition_config.stationary_frames,
        max_yaw_spread_rad=estimator_config.gates.max_yaw_spread_rad,
        max_plane_height_spread_m=estimator_config.gates.max_plane_p95_residual_m,
        metric_outer_size_m=estimator_config.geometry.outer_size_m,
        max_metric_center_spread_m=estimator_config.gates.max_center_spread_m,
    )
    hole_gate = StationaryHoleGate(
        required_frames=5,
        minimum_duration_s=acquisition_config.settle_duration_s,
        max_center_spread_m=estimator_config.gates.max_center_spread_m,
        max_yaw_spread_rad=estimator_config.gates.max_yaw_spread_rad,
    )
    held_hint = _replay_held_hint(root)
    requested_count = (
        len(reader) if max_frames is None else min(len(reader), int(max_frames))
    )

    writer: Any | None = None
    cv2 = None
    if mp4_path is not None:
        cv2 = _cv2()
        width = reader.metadata.depth_profile.intrinsics.width
        height = reader.metadata.depth_profile.intrinsics.height
        fps = float(reader.metadata.depth_profile.intrinsics.fps)
        writer = cv2.VideoWriter(
            str(mp4_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"cannot open MP4 writer: {mp4_path}")

    observations: list[PalletSceneObservation] = []
    frame_rows: list[dict[str, Any]] = []
    acquisition_rows: list[dict[str, Any]] = []
    timestamps: list[float] = []
    latencies_ms: list[float] = []
    try:
        for frame_index, frame in enumerate(reader):
            if frame_index >= requested_count:
                break
            timestamp_s = float(frame.depth_timestamp_ms) / 1_000.0
            start = time.perf_counter()
            observation = estimator.estimate(
                frame.depth_m(reader.metadata.depth_scale_m),
                reader.metadata.depth_profile.intrinsics,
                T_base_depth,
                timestamp_s=timestamp_s,
                frame_id=frame_index,
                color_on_depth_bgr=frame.color_on_depth_bgr,
                held_box_hint=held_hint,
                calibration_status="nominal_ready_assumed",
            )
            latency_ms = 1_000.0 * (time.perf_counter() - start)
            observations.append(observation)
            timestamps.append(timestamp_s)
            latencies_ms.append(latency_ms)
            in_audit_interval = (
                ACQUISITION_AUDIT_FIRST_FRAME
                <= frame_index
                <= ACQUISITION_AUDIT_LAST_FRAME
            )
            l_corner_status: LCornerGateStatus | None = None
            # The complete-hole dwell is audited over the whole recording so
            # short arrived sessions can demonstrate the fine-handoff evidence.
            # This remains perception-only: no odometry or wheel-stop evidence
            # exists in the recording.
            hole_status = hole_gate.update(observation, stationary=True)
            if in_audit_interval:
                # The recording has no wheel or odometry channel.  ``stationary=True``
                # is an explicit reviewed-replay assumption used only to audit
                # perception persistence; it never authorizes or simulates motion.
                l_corner_status = l_corner_gate.update(
                    observation.coarse,
                    stationary=True,
                )
            elif frame_index == ACQUISITION_AUDIT_LAST_FRAME + 1:
                l_corner_gate.clear()

            audit_phase = (
                "warmup"
                if ACQUISITION_AUDIT_FIRST_FRAME
                <= frame_index
                <= ACQUISITION_AUDIT_WARMUP_LAST_FRAME
                else "evaluated"
                if ACQUISITION_AUDIT_FIRST_EVALUATED_FRAME
                <= frame_index
                <= ACQUISITION_AUDIT_LAST_FRAME
                else "outside_fixed_interval"
            )
            would_request_metric_handoff = bool(
                l_corner_status is not None
                and l_corner_status.metric_proxy_stable
            )
            would_request_step = bool(
                audit_phase == "evaluated"
                and l_corner_status is not None
                and l_corner_status.stable
                and not would_request_metric_handoff
                and (hole_status is None or not hole_status.dwell_complete)
            )
            acquisition_audit = {
                "frame_ordinal_zero_based": frame_index,
                "in_fixed_review_interval": in_audit_interval,
                "phase": audit_phase,
                "stationary_input": {
                    "l_corner": (
                        "assumed_true_for_fixed_reviewed_interval"
                        if in_audit_interval
                        else "not_evaluated"
                    ),
                    "complete_hole": "assumed_true_for_replay_dwell_audit",
                },
                "l_corner_raw_valid": bool(
                    observation.coarse is not None and observation.coarse.valid
                ),
                "forward_acquisition_raw_valid": bool(
                    observation.coarse is not None
                    and observation.coarse.forward_acquisition_valid
                ),
                "complete_hole_raw_valid": bool(observation.valid),
                "l_corner_gate": _gate_payload(l_corner_status),
                "complete_hole_gate": _gate_payload(hole_status),
                "would_request_metric_proxy_handoff": (
                    would_request_metric_handoff
                ),
                "would_request_forward_step_from_geometry_only": would_request_step,
                "odometry": {
                    "available": False,
                    "status": "unavailable_recording_has_no_odometry_channel",
                },
                "motion_authorized": False,
                "offline_velocity_record_mps": {
                    "vx": 0.0,
                    "vy": 0.0,
                    "wz": 0.0,
                },
                "executed_forward_distance_m": 0.0,
                "execution_status": "offline_replay_no_robot_commands",
            }
            acquisition_rows.append(acquisition_audit)
            frame_row = {
                "frame_id": frame_index,
                "depth_frame_number": frame.depth_frame_number,
                "timestamp_s": timestamp_s,
                "latency_ms": latency_ms,
                "observation": observation.to_dict(),
                "acquisition_audit": acquisition_audit,
            }
            frame_rows.append(to_jsonable(frame_row))
            if writer is not None:
                image = (
                    frame.color_on_depth_bgr
                    if frame.color_on_depth_bgr is not None
                    else frame.raw_color_bgr
                )
                overlay = draw_pallet_overlay(
                    image,
                    observation,
                    evidence=estimator.last_evidence,
                    intrinsics=reader.metadata.depth_profile.intrinsics,
                    T_base_depth=T_base_depth,
                    frame_id=frame_index,
                    state="REPLAY",
                    latency_ms=latency_ms,
                    l_corner_gate=l_corner_status,
                    hole_gate=hole_status,
                    acquisition_audit=acquisition_audit,
                    slot1_hole_reference=slot1_hole_reference,
                )
                writer.write(overlay)
    finally:
        if writer is not None:
            writer.release()

    if jsonl_path is not None:
        jsonl_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in frame_rows
            ),
            encoding="utf-8",
        )
    if config_snapshot_path is not None:
        config_snapshot_path.write_text(
            json.dumps(to_jsonable(root), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    accepted = [item.stack for item in observations if item.valid]
    validity = [item.valid for item in observations]
    rejected_without_reason = sum(
        1
        for item in observations
        if not item.valid and not item.stack.rejection_reasons
    )
    rejection_histogram = Counter(
        reason
        for item in observations
        if not item.valid
        for reason in item.stack.rejection_reasons
    )
    centers = (
        np.stack([item.center_base for item in accepted])
        if accepted
        else np.empty((0, 3), dtype=np.float64)
    )
    yaws = [float(item.yaw_base_rad) for item in accepted]
    branches = [item.axis_branch for item in accepted]
    branch_flips = sum(
        previous != current for previous, current in zip(branches, branches[1:])
    )
    branch_right_ratio = (
        float(sum(branch == "image_right" for branch in branches) / len(branches))
        if branches
        else 0.0
    )
    consecutive_jump_values = [
        float(
            np.linalg.norm(
                current.stack.center_base[:2] - previous.stack.center_base[:2]
            )
        )
        for previous, current in zip(observations, observations[1:])
        if previous.valid and current.valid
    ]
    jumps = np.asarray(consecutive_jump_values, dtype=np.float64)
    opening_errors = [
        max(
            item.quality.get("opening_u_error_m", math.inf),
            item.quality.get("opening_v_error_m", math.inf),
        )
        for item in accepted
    ]
    plane_residuals = [
        item.quality.get("stack_plane_p95_residual_m", math.inf) for item in accepted
    ]
    orthogonality = [
        item.quality.get("orthogonality_error_rad", math.inf) for item in accepted
    ]
    stable_window = _has_stable_window(
        observations,
        length=5,
        center_spread_m=estimator_config.gates.max_center_spread_m,
        yaw_spread_rad=estimator_config.gates.max_yaw_spread_rad,
    )
    processed_ratio = float(len(observations) / max(requested_count, 1))
    valid_ratio = float(len(accepted) / max(len(observations), 1))
    best_one_second_ratio = _best_one_second_valid_ratio(timestamps, validity)
    center_std = np.std(centers, axis=0).tolist() if accepted else None
    yaw_std = _angle_std_rad(yaws)
    maximum_jump = float(np.max(jumps)) if jumps.size else None
    maximum_opening_error = max(opening_errors) if opening_errors else None
    maximum_plane_p95 = max(plane_residuals) if plane_residuals else None
    maximum_orthogonality = max(orthogonality) if orthogonality else None
    clearance_samples = [
        (
            float(item.held_top.top_plane_z_base_m),
            float(item.held_top.top_plane_z_uncertainty_m),
            float(item.stack.plane_height_base_m),
            float(item.stack.quality["stack_plane_p95_residual_m"]),
        )
        for item in observations
        if item.valid
        and item.held_top is not None
        and item.held_top.valid
        and item.held_top.distinct_from_stack
        and item.held_top.top_plane_z_base_m is not None
        and item.held_top.top_plane_z_uncertainty_m is not None
        and item.stack.plane_height_base_m is not None
        and "stack_plane_p95_residual_m" in item.stack.quality
    ]
    stack_top_samples = [
        (
            float(item.stack.plane_height_base_m),
            float(item.stack.quality["stack_plane_p95_residual_m"]),
        )
        for item in observations
        if item.valid
        and item.stack.plane_height_base_m is not None
        and "stack_plane_p95_residual_m" in item.stack.quality
    ]
    held_config = root.get("held_box", {})
    safety_config = root.get("safety", {})
    maximum_box_height_m = _finite_float(
        held_config.get("maximum_height_m", 0.164)
        if isinstance(held_config, Mapping)
        else 0.164,
        name="held_box.maximum_height_m",
    )
    if maximum_box_height_m <= 0.0:
        raise ValueError("held_box.maximum_height_m must be positive")
    minimum_clearance_m = _finite_float(
        safety_config.get("minimum_clearance_m", 0.050)
        if isinstance(safety_config, Mapping)
        else 0.050,
        name="safety.minimum_clearance_m",
    )
    if minimum_clearance_m <= 0.0:
        raise ValueError("safety.minimum_clearance_m must be positive")
    clearance_lower_bound_m = None
    direct_held_lower_m = None
    direct_stack_upper_m = None
    stack_upper_m = None
    valid_clearance_sample_count = 0
    valid_stack_top_sample_count = 0
    if clearance_samples:
        clearance_array = np.asarray(clearance_samples, dtype=np.float64)
        clearance_array = clearance_array[
            np.all(np.isfinite(clearance_array), axis=1)
        ]
        if clearance_array.size:
            valid_clearance_sample_count = int(clearance_array.shape[0])
            direct_held_lower_m = float(
                np.min(clearance_array[:, 0] - clearance_array[:, 1])
            )
            direct_stack_upper_m = float(
                np.max(clearance_array[:, 2] + clearance_array[:, 3])
            )
            clearance_lower_bound_m = (
                direct_held_lower_m - maximum_box_height_m - direct_stack_upper_m
            )
    if stack_top_samples:
        stack_top_array = np.asarray(stack_top_samples, dtype=np.float64)
        stack_top_array = stack_top_array[
            np.all(np.isfinite(stack_top_array), axis=1)
        ]
        if stack_top_array.size:
            valid_stack_top_sample_count = int(stack_top_array.shape[0])
            stack_upper_m = float(
                np.max(stack_top_array[:, 0] + stack_top_array[:, 1])
            )
    fixed_ready_clearance = _fixed_ready_clearance_audit(
        root,
        stack_upper_bound_m=stack_upper_m,
        maximum_box_height_m=maximum_box_height_m,
        minimum_clearance_m=minimum_clearance_m,
    )

    reviewed_acquisition_rows = [
        row for row in acquisition_rows if row["in_fixed_review_interval"]
    ]
    evaluated_acquisition_rows = [
        row for row in acquisition_rows if row["phase"] == "evaluated"
    ]
    stable_l_corner_rows = [
        row
        for row in evaluated_acquisition_rows
        if row["l_corner_gate"] is not None and bool(row["l_corner_gate"]["stable"])
    ]
    stable_metric_proxy_rows = [
        row
        for row in evaluated_acquisition_rows
        if row["l_corner_gate"] is not None
        and bool(row["l_corner_gate"]["metric_proxy_stable"])
    ]
    complete_hole_rows = [
        row
        for row in acquisition_rows
        if row["complete_hole_gate"] is not None
        and bool(row["complete_hole_gate"]["dwell_complete"])
    ]
    fixed_interval_complete = (
        len(reviewed_acquisition_rows)
        == ACQUISITION_AUDIT_LAST_FRAME - ACQUISITION_AUDIT_FIRST_FRAME + 1
        and len(evaluated_acquisition_rows)
        == ACQUISITION_AUDIT_EXPECTED_EVALUATED_FRAMES
    )
    acquisition_acceptance_checks = {
        "fixed_interval_72_through_105_present": fixed_interval_complete,
        "warmup_72_through_75_reserved": bool(
            len([row for row in reviewed_acquisition_rows if row["phase"] == "warmup"])
            == 4
        ),
        "stable_l_corner_frames_ge_24_of_30": (
            len(stable_l_corner_rows) >= ACQUISITION_AUDIT_REQUIRED_STABLE_FRAMES
        ),
        "offline_motion_authorized_false": not any(
            bool(row["motion_authorized"]) for row in acquisition_rows
        ),
        "offline_velocity_record_exact_zero": all(
            row["offline_velocity_record_mps"] == {"vx": 0.0, "vy": 0.0, "wz": 0.0}
            for row in acquisition_rows
        ),
        "odometry_never_claimed_available": not any(
            bool(row["odometry"]["available"]) for row in acquisition_rows
        ),
    }
    clearance_gate_passed = bool(
        clearance_lower_bound_m is not None
        and clearance_lower_bound_m >= minimum_clearance_m
    )
    acquisition_blockers = ["recording_has_no_odometry_channel"]
    if acquisition_config.budget_m <= 0.0:
        acquisition_blockers.append("configured_forward_budget_is_zero")
    if not clearance_gate_passed:
        acquisition_blockers.append("vertical_clearance_gate_not_passed")
    acquisition_summary = {
        "schema_version": 1,
        "purpose": "reviewed_perception_audit_not_motion_replay",
        "frame_ordinals": {
            "ordering": "session_reader_sorted_zero_based",
            "reviewed_inclusive": [
                ACQUISITION_AUDIT_FIRST_FRAME,
                ACQUISITION_AUDIT_LAST_FRAME,
            ],
            "warmup_inclusive": [
                ACQUISITION_AUDIT_FIRST_FRAME,
                ACQUISITION_AUDIT_WARMUP_LAST_FRAME,
            ],
            "evaluated_inclusive": [
                ACQUISITION_AUDIT_FIRST_EVALUATED_FRAME,
                ACQUISITION_AUDIT_LAST_FRAME,
            ],
            "reviewed_count": len(reviewed_acquisition_rows),
            "evaluated_count": len(evaluated_acquisition_rows),
        },
        "stationary_assumption": {
            "l_corner": "true_only_for_fixed_frames_72_through_105",
            "complete_hole": "true_for_full_replay_dwell_audit",
            "control_authority": "none_offline_replay",
        },
        "l_corner": {
            "raw_valid_reviewed_count": sum(
                bool(row["l_corner_raw_valid"]) for row in reviewed_acquisition_rows
            ),
            "forward_acquisition_raw_valid_reviewed_count": sum(
                bool(row["forward_acquisition_raw_valid"])
                for row in reviewed_acquisition_rows
            ),
            "stable_evaluated_count": len(stable_l_corner_rows),
            "metric_proxy_stable_evaluated_count": len(
                stable_metric_proxy_rows
            ),
            "required_stable_count": ACQUISITION_AUDIT_REQUIRED_STABLE_FRAMES,
            "expected_evaluated_count": ACQUISITION_AUDIT_EXPECTED_EVALUATED_FRAMES,
            "stable_frame_ordinals": [
                row["frame_ordinal_zero_based"] for row in stable_l_corner_rows
            ],
            "metric_proxy_stable_frame_ordinals": [
                row["frame_ordinal_zero_based"]
                for row in stable_metric_proxy_rows
            ],
        },
        "complete_hole": {
            "raw_valid_recording_count": sum(
                bool(row["complete_hole_raw_valid"]) for row in acquisition_rows
            ),
            "raw_valid_reviewed_interval_count": sum(
                bool(row["complete_hole_raw_valid"])
                for row in reviewed_acquisition_rows
            ),
            "dwell_complete_frame_ordinals": [
                row["frame_ordinal_zero_based"] for row in complete_hole_rows
            ],
        },
        "geometry_only_would_request_step_count": sum(
            bool(row["would_request_forward_step_from_geometry_only"])
            for row in evaluated_acquisition_rows
        ),
        "geometry_only_would_request_metric_proxy_handoff_count": sum(
            bool(row["would_request_metric_proxy_handoff"])
            for row in evaluated_acquisition_rows
        ),
        "configured_limits": {
            "forward_budget_m": acquisition_config.budget_m,
            "forward_step_m": acquisition_config.step_m,
            "forward_speed_mps": acquisition_config.speed_mps,
        },
        "odometry": {
            "available": False,
            "status": "unavailable_recording_has_no_odometry_channel",
        },
        "motion_authorized": False,
        "mobile_actuation_count": 0,
        "executed_forward_distance_m": 0.0,
        "fine_controller_transition": (
            "metric_proxy_handoff_would_be_requested_offline_not_executed"
            if stable_metric_proxy_rows
            else "not_executed_offline_replay"
        ),
        "motion_blockers": acquisition_blockers,
        "acceptance": {
            "applicable": fixed_interval_complete,
            "passed": (
                bool(all(acquisition_acceptance_checks.values()))
                if fixed_interval_complete
                else None
            ),
            "status": (
                "pass"
                if fixed_interval_complete
                and all(acquisition_acceptance_checks.values())
                else "fail"
                if fixed_interval_complete
                else "not_applicable_fixed_interval_absent"
            ),
            "checks": acquisition_acceptance_checks,
        },
    }
    acceptance = _session_acceptance(
        session_name,
        processed_ratio=processed_ratio,
        valid_ratio=valid_ratio,
        rejected_without_reason=rejected_without_reason,
        branch_flips=branch_flips,
        branch_image_right_ratio=branch_right_ratio,
        center_std_xyz_m=center_std,
        yaw_std_rad=yaw_std,
        stable_window=stable_window,
        best_one_second_ratio=best_one_second_ratio,
        maximum_center_jump_m=maximum_jump,
        maximum_opening_error_m=maximum_opening_error,
        maximum_plane_p95_m=maximum_plane_p95,
        maximum_orthogonality_rad=maximum_orthogonality,
        config=estimator_config,
    )
    latency = np.asarray(latencies_ms, dtype=np.float64)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "session": str(Path(session_path)),
        "session_name": session_name,
        "recorded_frame_count": len(reader),
        "requested_frame_count": requested_count,
        "processed_frame_count": len(observations),
        "processed_frame_ratio": processed_ratio,
        "valid_frame_count": len(accepted),
        "valid_ratio": valid_ratio,
        "rejection_reason_histogram": dict(sorted(rejection_histogram.items())),
        "rejected_without_reason_count": rejected_without_reason,
        "center_base_mean_xyz_m": np.mean(centers, axis=0).tolist()
        if accepted
        else None,
        "center_base_std_xyz_m": center_std,
        "yaw_base_mean_deg": (
            math.degrees(float(np.mean(np.unwrap(np.asarray(yaws))))) if yaws else None
        ),
        "yaw_base_std_deg": None if yaw_std is None else math.degrees(yaw_std),
        "axis_branch_flip_count": branch_flips,
        "image_right_branch_ratio": branch_right_ratio,
        "longest_consecutive_valid_frames": _longest_true_run(validity),
        "stable_five_frame_window": stable_window,
        "best_one_second_valid_ratio": best_one_second_ratio,
        "maximum_consecutive_accepted_center_jump_m": maximum_jump,
        "maximum_accepted_opening_error_m": maximum_opening_error,
        "maximum_accepted_plane_p95_residual_m": maximum_plane_p95,
        "maximum_accepted_orthogonality_error_deg": (
            None
            if maximum_orthogonality is None
            else math.degrees(maximum_orthogonality)
        ),
        "latency_ms": {
            "mean": float(np.mean(latency)) if latency.size else None,
            "p50": float(np.percentile(latency, 50)) if latency.size else None,
            "p95": float(np.percentile(latency, 95)) if latency.size else None,
            "maximum": float(np.max(latency)) if latency.size else None,
        },
        "registration": registration,
        "slot1_hole_reference": (
            None
            if slot1_hole_reference is None
            else slot1_hole_reference.to_dict()
        ),
        "profile": profile,
        "acceptance": acceptance,
        "acquisition_audit": acquisition_summary,
        "absolute_placement_accuracy": "not_measured_no_external_ground_truth",
        "vertical_clearance": {
            "valid_held_top_frame_count": valid_clearance_sample_count,
            "valid_stack_top_frame_count": valid_stack_top_sample_count,
            "maximum_box_height_m": maximum_box_height_m,
            "required_lower_bound_m": minimum_clearance_m,
            "observed_conservative_lower_bound_m": clearance_lower_bound_m,
            "direct_depth_held_top": {
                "source": "direct_held_top_minus_max_box_height_to_stack_plane",
                "valid_held_top_frame_count": valid_clearance_sample_count,
                "held_top_lower_z_base_m": direct_held_lower_m,
                "conservative_stack_top_z_base_m": direct_stack_upper_m,
                "required_lower_bound_m": minimum_clearance_m,
                "observed_conservative_lower_bound_m": clearance_lower_bound_m,
                "gate_passed": bool(clearance_gate_passed),
                "status": (
                    "unobservable"
                    if clearance_lower_bound_m is None
                    else "pass"
                    if clearance_lower_bound_m >= minimum_clearance_m
                    else "fail_raise_ready_pose"
                ),
            },
            "fixed_ready_configured_geometry": fixed_ready_clearance,
            "nonzero_motion_gate_passed": bool(clearance_gate_passed),
            "motion_authorized_from_nominal_geometry": False,
            "status": (
                "unobservable"
                if clearance_lower_bound_m is None
                else "pass"
                if clearance_lower_bound_m >= minimum_clearance_m
                else "fail_raise_ready_pose"
            ),
        },
        "artifacts": {
            "mp4": None if mp4_path is None else str(mp4_path),
            "summary": None if summary_path is None else str(summary_path),
            "jsonl": None if jsonl_path is None else str(jsonl_path),
            "config_snapshot": (
                None if config_snapshot_path is None else str(config_snapshot_path)
            ),
        },
    }
    summary = to_jsonable(summary)
    if summary_path is not None:
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return summary


replay_pallet_session = evaluate_pallet_session


__all__ = ["evaluate_pallet_session", "replay_pallet_session"]
