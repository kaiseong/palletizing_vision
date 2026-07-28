"""Deterministic estimator-only benchmark and behavioral-equivalence gate.

The runner preloads recorded depth frames, excludes file I/O and rendering from
timing, and keeps one estimator alive across warmup and measured passes to
match the intended live-perception lifecycle.  It proves parity with a prior
unlabelled recording result; it does not prove physical pose accuracy.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .calibration import load_calibration, load_json
from .estimator import ParcelPoseEstimator
from .evaluation import base_pose_from_estimate
from .models import EstimatorConfig
from .output import pose_estimate_to_dict, to_jsonable
from .recording import MANIFEST_NAME, SessionReader


_THREAD_ENVIRONMENT_KEYS = (
    "PYTHONHASHSEED",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_REQUIRED_FIXTURE_HASH_KEYS = (
    "manifest_sha256",
    "calibration_sha256",
    "config_sha256",
)
_REQUIRED_SETTING_KEYS = (
    "warmup_passes",
    "repeats",
    "estimator_lifecycle",
    "timed_scope",
    "depth_preloaded",
    "gc_disabled_during_measured_pass",
    "opencv_opencl_enabled",
)
_PUBLIC_CENTER_FIELDS = (
    "center_plane_xy_m",
    "center_depth_m",
    "center_base_xy_m",
    "top_center_base_xyz_m",
    "box_center_base_xyz_m",
)
_PUBLIC_AXIS_FIELDS = (
    "long_axis_plane_xy",
    "short_axis_plane_xy",
    "long_axis_base_xy",
    "short_axis_base_xy",
)
_PUBLIC_LINE_RAD_FIELDS = ("yaw_rad",)
_PUBLIC_LINE_DEG_FIELDS = ("yaw_mod_180_deg", "long_axis_yaw_base_deg")
_PUBLIC_SCALAR_DEG_FIELDS = (
    "canonical_residual_deg",
    "classification_margin_deg",
)
_PUBLIC_EXACT_FIELDS = (
    "timestamp_ms",
    "frame_id",
    "frame",
    "box_model_m",
    "canonical_reference_deg",
    "observability",
    "calibration",
)
_PUBLIC_REQUIRED_FIELDS = frozenset(
    {
        *_PUBLIC_EXACT_FIELDS,
        "center_plane_xy_m",
        "center_depth_m",
        "yaw_rad",
        "yaw_mod_180_deg",
        "canonical_residual_deg",
        "classification_margin_deg",
        "long_axis_plane_xy",
        "short_axis_plane_xy",
        "center_feasible_set",
        "confidence",
    }
)
_PUBLIC_BASE_FIELDS = frozenset(
    {
        "center_base_xy_m",
        "top_center_base_xyz_m",
        "box_center_base_xyz_m",
        "long_axis_base_xy",
        "short_axis_base_xy",
        "long_axis_yaw_base_deg",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_source_sha256() -> str:
    package_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package_root.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _hardware_identity() -> dict[str, str | None]:
    device_model: str | None = None
    for path in (
        Path("/sys/firmware/devicetree/base/model"),
        Path("/proc/device-tree/model"),
    ):
        try:
            device_model = path.read_bytes().rstrip(b"\0").decode("utf-8")
            break
        except (OSError, UnicodeDecodeError):
            continue
    cpu_model: str | None = None
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip().lower() in {"model name", "hardware"}:
                cpu_model = value.strip()
                break
    except OSError:
        pass
    return {"cpu_model": cpu_model, "device_model": device_model}


def _line_angle_error_deg(first: float, second: float) -> float:
    return abs((float(first) - float(second) + 90.0) % 180.0 - 90.0)


def _latency_summary(per_frame_ms: np.ndarray, pass_totals_ms: np.ndarray) -> dict[str, Any]:
    frame_medians = np.median(per_frame_ms, axis=0)
    pass_p95 = np.percentile(per_frame_ms, 95, axis=1)
    estimator_totals = np.sum(per_frame_ms, axis=1)
    pass_throughput = 1000.0 * per_frame_ms.shape[1] / estimator_totals
    p95_median = float(np.median(pass_p95))
    p95_mad = float(np.median(np.abs(pass_p95 - p95_median)))
    return {
        "frame_count": int(per_frame_ms.shape[1]),
        "repeat_count": int(per_frame_ms.shape[0]),
        "per_frame_median_ms": {
            "mean": float(np.mean(frame_medians)),
            "p50": float(np.percentile(frame_medians, 50)),
            "p95": float(np.percentile(frame_medians, 95)),
            "maximum": float(np.max(frame_medians)),
        },
        "per_pass": {
            "estimator_total_ms": estimator_totals.tolist(),
            "wall_total_ms": pass_totals_ms.tolist(),
            "p95_ms": pass_p95.tolist(),
            "throughput_fps": pass_throughput.tolist(),
            "median_throughput_fps": float(np.median(pass_throughput)),
            "p95_mad_ratio": p95_mad / max(abs(p95_median), 1e-12),
        },
    }


def _runtime_environment(cv2: Any) -> dict[str, Any]:
    affinity = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None
    )
    return {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": str(np.__version__),
        "opencv_version": str(cv2.__version__),
        "opencv_threads": int(cv2.getNumThreads()),
        "cpu_affinity": affinity,
        "hardware_identity": _hardware_identity(),
        "thread_environment": {
            key: os.environ.get(key) for key in _THREAD_ENVIRONMENT_KEYS
        },
    }


def _environment_fingerprint(environment: Mapping[str, Any]) -> str:
    payload = json.dumps(environment, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _estimator_config(config_payload: Mapping[str, Any]) -> EstimatorConfig:
    return EstimatorConfig.from_root_config(config_payload)


def _frame_signature(
    frame_index: int,
    estimate: Any,
    calibration: Any,
) -> dict[str, Any]:
    pose = base_pose_from_estimate(estimate, calibration)
    public_pose = pose_estimate_to_dict(estimate)
    public_pose.pop("diagnostics", None)
    return {
        "frame_index": int(frame_index),
        "frame_id": int(estimate.frame_id),
        "timestamp_ms": float(estimate.timestamp_ms),
        "full_pose_valid": bool(estimate.full_pose_valid),
        "absolute_valid": bool(estimate.absolute_valid),
        "base_registration": str(estimate.base_registration),
        "canonical_reference_deg": estimate.canonical_reference_deg,
        "observability": dict(estimate.observability),
        "reasons": list(estimate.reasons),
        "yaw_confidence": float(estimate.per_field_confidence.get("yaw", 0.0)),
        "public_pose": public_pose,
        "box_center_base_xyz_m": (
            None if pose is None else list(pose.box_center_xyz_m)
        ),
        "top_center_base_xyz_m": (
            None if pose is None else list(pose.top_center_xyz_m)
        ),
        "yaw_base_signed_deg": None if pose is None else pose.yaw_signed_deg,
    }


def _signature_digest(signatures: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        to_jsonable(signatures),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def benchmark_session(
    session: str | Path,
    calibration_path: str | Path,
    config_path: str | Path,
    *,
    warmup_passes: int = 1,
    repeats: int = 5,
) -> dict[str, Any]:
    """Benchmark one fixed recording with I/O and rendering outside timing."""

    if warmup_passes < 0 or repeats < 1:
        raise ValueError("warmup_passes must be >= 0 and repeats must be >= 1")
    session_path = Path(session)
    calibration_file = Path(calibration_path)
    config_file = Path(config_path)
    reader = SessionReader(session_path)
    if len(reader) == 0:
        raise ValueError("benchmark session contains no frames")

    depth_frames: list[tuple[np.ndarray, float, int]] = []
    for frame in reader:
        depth_frames.append(
            (
                np.ascontiguousarray(frame.raw_depth_z16),
                float(frame.depth_timestamp_ms),
                int(frame.depth_frame_number),
            )
        )

    calibration = load_calibration(calibration_file)
    config = _estimator_config(load_json(config_file))
    estimator = ParcelPoseEstimator(
        reader.metadata.depth_profile.intrinsics,
        calibration,
        config,
    )

    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for the estimator benchmark") from exc
    cv2.setNumThreads(1)
    if hasattr(cv2, "ocl"):
        cv2.ocl.setUseOpenCL(False)

    def estimate(item: tuple[np.ndarray, float, int]) -> Any:
        depth, timestamp_ms, frame_id = item
        return estimator.estimate(
            depth,
            depth_scale=reader.metadata.depth_scale_m,
            timestamp_ms=timestamp_ms,
            frame_id=frame_id,
        )

    for _ in range(warmup_passes):
        for item in depth_frames:
            estimate(item)

    measured = np.empty((repeats, len(depth_frames)), dtype=np.float64)
    pass_totals = np.empty(repeats, dtype=np.float64)
    reference_signatures: list[dict[str, Any]] | None = None
    reference_digest: str | None = None
    repeat_digests: list[str] = []
    for repeat_index in range(repeats):
        gc.collect()
        gc_was_enabled = gc.isenabled()
        gc.disable()
        signatures: list[dict[str, Any]] = []
        pass_start = time.perf_counter_ns()
        try:
            for frame_index, item in enumerate(depth_frames):
                start = time.perf_counter_ns()
                result = estimate(item)
                measured[repeat_index, frame_index] = (
                    time.perf_counter_ns() - start
                ) / 1_000_000.0
                signatures.append(_frame_signature(frame_index, result, calibration))
        finally:
            pass_totals[repeat_index] = (
                time.perf_counter_ns() - pass_start
            ) / 1_000_000.0
            if gc_was_enabled:
                gc.enable()
        digest = _signature_digest(signatures)
        repeat_digests.append(digest)
        if reference_signatures is None:
            reference_signatures = signatures
            reference_digest = digest

    if reference_signatures is None or reference_digest is None:
        raise RuntimeError("benchmark did not produce frame signatures")

    environment = _runtime_environment(cv2)
    latency = _latency_summary(measured, pass_totals)
    stable_outputs = len(set(repeat_digests)) == 1
    stable_timing = float(latency["per_pass"]["p95_mad_ratio"]) <= 0.03
    valid_count = sum(
        bool(frame["full_pose_valid"]) for frame in reference_signatures
    )
    return to_jsonable(
        {
            "schema_version": 1,
            "kind": "parcel_pose_estimator_benchmark",
            "fixture": {
                "session": str(session_path),
                "manifest_sha256": _sha256(session_path / MANIFEST_NAME),
                "calibration": str(calibration_file),
                "calibration_sha256": _sha256(calibration_file),
                "config": str(config_file),
                "config_sha256": _sha256(config_file),
            },
            "settings": {
                "warmup_passes": warmup_passes,
                "repeats": repeats,
                "estimator_lifecycle": "one instance across warmup and measured passes",
                "timed_scope": "ParcelPoseEstimator.estimate only",
                "depth_preloaded": True,
                "gc_disabled_during_measured_pass": True,
                "opencv_opencl_enabled": False,
            },
            "environment": environment,
            "environment_fingerprint": _environment_fingerprint(environment),
            "source_provenance": {
                "package_python_sha256": _package_source_sha256(),
            },
            "behavior": {
                "frame_count": len(reference_signatures),
                "full_pose_valid_frames": valid_count,
                "abstained_frames": len(reference_signatures) - valid_count,
                "absolute_base_pose_frames": sum(
                    bool(frame["absolute_valid"]) for frame in reference_signatures
                ),
                "signature_sha256": reference_digest,
                "repeat_signature_sha256": repeat_digests,
                "repeat_outputs_identical": stable_outputs,
                "frames": reference_signatures,
            },
            "latency": latency,
            "stability": {
                "outputs_identical": stable_outputs,
                "p95_mad_ratio_within_3_percent": stable_timing,
            },
        }
    )


def _numeric_error_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"p95": 0.0, "maximum": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _is_optional_finite_number(value: Any) -> bool:
    return value is None or _is_finite_number(value)


def _is_optional_vector(value: Any, length: int) -> bool:
    return value is None or (
        isinstance(value, (list, tuple))
        and len(value) == length
        and all(_is_finite_number(item) for item in value)
    )


def _is_string_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and all(
        isinstance(key, str) and isinstance(item, str)
        for key, item in value.items()
    )


def _is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _public_pose_validation_errors(
    pose: Mapping[str, Any],
    *,
    label: str,
) -> list[str]:
    errors: list[str] = []
    for field in sorted(_PUBLIC_REQUIRED_FIELDS - set(pose)):
        errors.append(f"{label}.{field}")

    if not _is_finite_number(pose.get("timestamp_ms")):
        errors.append(f"{label}.timestamp_ms")
    frame_id = pose.get("frame_id")
    if not isinstance(frame_id, int) or isinstance(frame_id, bool):
        errors.append(f"{label}.frame_id")
    frame = pose.get("frame")
    if not isinstance(frame, str) or not frame:
        errors.append(f"{label}.frame")

    box_model = pose.get("box_model_m")
    if not isinstance(box_model, Mapping) or not all(
        _is_finite_number(box_model.get(field))
        and float(box_model[field]) > 0.0
        for field in ("long", "short", "height")
    ):
        errors.append(f"{label}.box_model_m")
    if not _is_optional_finite_number(pose.get("canonical_reference_deg")):
        errors.append(f"{label}.canonical_reference_deg")
    if not _is_string_mapping(pose.get("observability")):
        errors.append(f"{label}.observability")

    vector_lengths = {
        "center_plane_xy_m": 2,
        "center_depth_m": 3,
        "center_base_xy_m": 2,
        "top_center_base_xyz_m": 3,
        "box_center_base_xyz_m": 3,
        "long_axis_plane_xy": 2,
        "short_axis_plane_xy": 2,
        "long_axis_base_xy": 2,
        "short_axis_base_xy": 2,
    }
    for field, length in vector_lengths.items():
        if field in pose and not _is_optional_vector(pose[field], length):
            errors.append(f"{label}.{field}")
    for field in (
        *_PUBLIC_LINE_RAD_FIELDS,
        *_PUBLIC_LINE_DEG_FIELDS,
        *_PUBLIC_SCALAR_DEG_FIELDS,
    ):
        if field in pose and not _is_optional_finite_number(pose[field]):
            errors.append(f"{label}.{field}")

    feasible = pose.get("center_feasible_set")
    if feasible is not None:
        if not isinstance(feasible, Mapping):
            errors.append(f"{label}.center_feasible_set")
        else:
            for axis, interval in feasible.items():
                if (
                    not isinstance(axis, str)
                    or interval is None
                    or not _is_optional_vector(interval, 2)
                ):
                    errors.append(f"{label}.center_feasible_set.{axis}")

    calibration = pose.get("calibration")
    if not isinstance(calibration, Mapping):
        errors.append(f"{label}.calibration")
    else:
        for field in ("state", "base_registration"):
            value = calibration.get(field)
            if not isinstance(value, str) or not value:
                errors.append(f"{label}.calibration.{field}")
        for field in (
            "base_registration_valid",
            "absolute_base_validated",
        ):
            if not isinstance(calibration.get(field), bool):
                errors.append(f"{label}.calibration.{field}")
        if calibration.get("base_registration_valid") is True:
            for field in sorted(_PUBLIC_BASE_FIELDS - set(pose)):
                errors.append(f"{label}.{field}")

    confidence = pose.get("confidence")
    if not isinstance(confidence, Mapping):
        errors.append(f"{label}.confidence")
    else:
        for field in (
            "geometry_valid",
            "full_pose_valid",
            "absolute_base_pose_valid",
        ):
            if not isinstance(confidence.get(field), bool):
                errors.append(f"{label}.confidence.{field}")
        if not _is_string_list(confidence.get("reasons")):
            errors.append(f"{label}.confidence.reasons")
        per_field = confidence.get("per_field")
        if not isinstance(per_field, Mapping):
            errors.append(f"{label}.confidence.per_field")
        else:
            for field, value in per_field.items():
                if (
                    not isinstance(field, str)
                    or not _is_finite_number(value)
                    or not 0.0 <= float(value) <= 1.0
                ):
                    errors.append(f"{label}.confidence.per_field.{field}")
    return errors


def _report_validation_errors(
    report: Mapping[str, Any],
    *,
    label: str,
) -> list[str]:
    errors: list[str] = []
    if report.get("schema_version") != 1:
        errors.append(f"{label}:schema_version")
    if report.get("kind") != "parcel_pose_estimator_benchmark":
        errors.append(f"{label}:kind")

    fixture = report.get("fixture")
    if not isinstance(fixture, Mapping):
        errors.append(f"{label}:fixture")
    else:
        for key in _REQUIRED_FIXTURE_HASH_KEYS:
            if not _is_sha256(fixture.get(key)):
                errors.append(f"{label}:fixture.{key}")

    settings = report.get("settings")
    repeat_count: int | None = None
    if not isinstance(settings, Mapping):
        errors.append(f"{label}:settings")
    else:
        for key in _REQUIRED_SETTING_KEYS:
            if key not in settings:
                errors.append(f"{label}:settings.{key}")
        warmup_passes = settings.get("warmup_passes")
        repeats = settings.get("repeats")
        if (
            not isinstance(warmup_passes, int)
            or isinstance(warmup_passes, bool)
            or warmup_passes < 0
        ):
            errors.append(f"{label}:settings.warmup_passes")
        if (
            not isinstance(repeats, int)
            or isinstance(repeats, bool)
            or repeats < 1
        ):
            errors.append(f"{label}:settings.repeats")
        else:
            repeat_count = repeats
        for key in ("estimator_lifecycle", "timed_scope"):
            value = settings.get(key)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{label}:settings.{key}")
        expected_boolean_settings = {
            "depth_preloaded": True,
            "gc_disabled_during_measured_pass": True,
            "opencv_opencl_enabled": False,
        }
        for key, expected in expected_boolean_settings.items():
            if settings.get(key) is not expected:
                errors.append(f"{label}:settings.{key}")

    environment = report.get("environment")
    fingerprint = report.get("environment_fingerprint")
    if not isinstance(environment, Mapping) or not environment:
        errors.append(f"{label}:environment")
    elif not _is_sha256(fingerprint) or fingerprint != _environment_fingerprint(
        environment
    ):
        errors.append(f"{label}:environment_fingerprint")

    source = report.get("source_provenance")
    if not isinstance(source, Mapping) or not _is_sha256(
        source.get("package_python_sha256")
    ):
        errors.append(f"{label}:source_provenance.package_python_sha256")

    behavior = report.get("behavior")
    frames: list[Any] | None = None
    if not isinstance(behavior, Mapping):
        errors.append(f"{label}:behavior")
    else:
        candidate_frames = behavior.get("frames")
        if not isinstance(candidate_frames, list):
            errors.append(f"{label}:behavior.frames")
        else:
            frames = candidate_frames
            if not frames or behavior.get("frame_count") != len(frames):
                errors.append(f"{label}:behavior.frame_count")
        signature = behavior.get("signature_sha256")
        if not _is_sha256(signature):
            errors.append(f"{label}:behavior.signature_sha256")
        elif frames is not None and signature != _signature_digest(frames):
            errors.append(f"{label}:behavior.signature_integrity")
        repeat_digests = behavior.get("repeat_signature_sha256")
        if not isinstance(repeat_digests, list) or not repeat_digests or not all(
            _is_sha256(item) for item in repeat_digests
        ):
            errors.append(f"{label}:behavior.repeat_signature_sha256")
        elif repeat_count is not None and len(repeat_digests) != repeat_count:
            errors.append(f"{label}:behavior.repeat_signature_count")
        repeat_outputs_identical = behavior.get("repeat_outputs_identical")
        if not isinstance(repeat_outputs_identical, bool):
            errors.append(f"{label}:behavior.repeat_outputs_identical")
        elif isinstance(repeat_digests, list) and repeat_digests:
            actually_identical = len(set(repeat_digests)) == 1
            if repeat_outputs_identical != actually_identical:
                errors.append(f"{label}:behavior.repeat_signature_consistency")
            if (
                actually_identical
                and _is_sha256(signature)
                and repeat_digests[0] != signature
            ):
                errors.append(f"{label}:behavior.repeat_signature_reference")
    if frames is not None:
        valid_count = 0
        absolute_count = 0
        for index, frame in enumerate(frames):
            if not isinstance(frame, Mapping):
                errors.append(f"{label}:behavior.frames[{index}]")
                continue
            if frame.get("frame_index") != index:
                errors.append(f"{label}:behavior.frames[{index}].frame_index")
            if not isinstance(frame.get("full_pose_valid"), bool):
                errors.append(f"{label}:behavior.frames[{index}].full_pose_valid")
            else:
                valid_count += int(frame["full_pose_valid"])
            if not isinstance(frame.get("absolute_valid"), bool):
                errors.append(f"{label}:behavior.frames[{index}].absolute_valid")
            else:
                absolute_count += int(frame["absolute_valid"])
            public_pose = frame.get("public_pose")
            if not isinstance(public_pose, Mapping):
                errors.append(f"{label}:behavior.frames[{index}].public_pose")
                continue
            public_label = f"{label}:behavior.frames[{index}].public_pose"
            errors.extend(
                _public_pose_validation_errors(public_pose, label=public_label)
            )
            if not _is_finite_number(frame.get("timestamp_ms")):
                errors.append(f"{label}:behavior.frames[{index}].timestamp_ms")
            frame_id = frame.get("frame_id")
            if not isinstance(frame_id, int) or isinstance(frame_id, bool):
                errors.append(f"{label}:behavior.frames[{index}].frame_id")
            if not isinstance(frame.get("base_registration"), str):
                errors.append(f"{label}:behavior.frames[{index}].base_registration")
            if not _is_optional_finite_number(
                frame.get("canonical_reference_deg")
            ):
                errors.append(
                    f"{label}:behavior.frames[{index}].canonical_reference_deg"
                )
            if not _is_string_mapping(frame.get("observability")):
                errors.append(f"{label}:behavior.frames[{index}].observability")
            if not _is_string_list(frame.get("reasons")):
                errors.append(f"{label}:behavior.frames[{index}].reasons")
            yaw_confidence = frame.get("yaw_confidence")
            if not (
                _is_finite_number(yaw_confidence)
                and 0.0 <= float(yaw_confidence) <= 1.0
            ):
                errors.append(f"{label}:behavior.frames[{index}].yaw_confidence")
            for field in (
                "box_center_base_xyz_m",
                "top_center_base_xyz_m",
            ):
                if not _is_optional_vector(frame.get(field), 3):
                    errors.append(f"{label}:behavior.frames[{index}].{field}")
            if not _is_optional_finite_number(frame.get("yaw_base_signed_deg")):
                errors.append(
                    f"{label}:behavior.frames[{index}].yaw_base_signed_deg"
                )

            confidence = public_pose.get("confidence")
            calibration = public_pose.get("calibration")
            if isinstance(confidence, Mapping):
                consistency = {
                    "full_pose_valid": frame.get("full_pose_valid"),
                    "absolute_base_pose_valid": frame.get("absolute_valid"),
                    "reasons": frame.get("reasons"),
                }
                for field, expected in consistency.items():
                    if confidence.get(field) != expected:
                        errors.append(
                            f"{label}:behavior.frames[{index}].{field}_consistency"
                        )
                per_field = confidence.get("per_field")
                if isinstance(per_field, Mapping) and per_field.get("yaw") != yaw_confidence:
                    errors.append(
                        f"{label}:behavior.frames[{index}].yaw_confidence_consistency"
                    )
            public_consistency = {
                "timestamp_ms": frame.get("timestamp_ms"),
                "frame_id": frame.get("frame_id"),
                "canonical_reference_deg": frame.get("canonical_reference_deg"),
                "observability": frame.get("observability"),
            }
            for field, expected in public_consistency.items():
                if public_pose.get(field) != expected:
                    errors.append(
                        f"{label}:behavior.frames[{index}].{field}_consistency"
                    )
            if (
                isinstance(calibration, Mapping)
                and calibration.get("base_registration")
                != frame.get("base_registration")
            ):
                errors.append(
                    f"{label}:behavior.frames[{index}].base_registration_consistency"
                )
        if behavior.get("full_pose_valid_frames") != valid_count:
            errors.append(f"{label}:behavior.full_pose_valid_frames")
        if behavior.get("abstained_frames") != len(frames) - valid_count:
            errors.append(f"{label}:behavior.abstained_frames")
        if behavior.get("absolute_base_pose_frames") != absolute_count:
            errors.append(f"{label}:behavior.absolute_base_pose_frames")

    latency = report.get("latency")
    computed_p95_mad_ratio: float | None = None
    try:
        if not isinstance(latency, Mapping):
            raise TypeError
        p50 = float(latency["per_frame_median_ms"]["p50"])
        p95 = float(latency["per_frame_median_ms"]["p95"])
        per_pass = latency["per_pass"]
        throughput = float(per_pass["median_throughput_fps"])
        stored_p95_mad_ratio = float(per_pass["p95_mad_ratio"])
        estimator_totals = [float(value) for value in per_pass["estimator_total_ms"]]
        wall_totals = [float(value) for value in per_pass["wall_total_ms"]]
        p95_values = [float(value) for value in per_pass["p95_ms"]]
        throughput_values = [float(value) for value in per_pass["throughput_fps"]]
        pass_lengths = {
            len(estimator_totals),
            len(wall_totals),
            len(p95_values),
            len(throughput_values),
        }
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (
                p50,
                p95,
                throughput,
                *estimator_totals,
                *wall_totals,
                *p95_values,
                *throughput_values,
            )
        ) or not (
            math.isfinite(stored_p95_mad_ratio)
            and stored_p95_mad_ratio >= 0.0
        ):
            raise ValueError
        if p95 < p50 or latency.get("frame_count") != len(frames or []):
            raise ValueError
        if repeat_count is not None and (
            latency.get("repeat_count") != repeat_count
            or pass_lengths != {repeat_count}
        ):
            raise ValueError
        computed_throughput = float(np.median(throughput_values))
        p95_median = float(np.median(p95_values))
        computed_p95_mad_ratio = float(
            np.median(np.abs(np.asarray(p95_values) - p95_median))
        ) / max(abs(p95_median), 1e-12)
        if not math.isclose(
            throughput,
            computed_throughput,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ) or not math.isclose(
            stored_p95_mad_ratio,
            computed_p95_mad_ratio,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError
    except (KeyError, TypeError, ValueError, OverflowError):
        errors.append(f"{label}:latency")

    stability = report.get("stability")
    if not isinstance(stability, Mapping) or not all(
        isinstance(stability.get(key), bool)
        for key in ("outputs_identical", "p95_mad_ratio_within_3_percent")
    ):
        errors.append(f"{label}:stability")
    elif isinstance(behavior, Mapping):
        if stability.get("outputs_identical") != behavior.get(
            "repeat_outputs_identical"
        ):
            errors.append(f"{label}:stability.outputs_identical_consistency")
        if computed_p95_mad_ratio is not None:
            expected_timing_stability = computed_p95_mad_ratio <= 0.03
            if (
                stability.get("p95_mad_ratio_within_3_percent")
                != expected_timing_stability
            ):
                errors.append(f"{label}:stability.p95_mad_ratio_consistency")
    return errors


def _append_vector_error(
    reference: Mapping[str, Any],
    current: Mapping[str, Any],
    field: str,
    *,
    frame_index: int,
    scale: float,
    structural_errors: list[str],
    errors: list[float],
) -> None:
    first = reference.get(field)
    second = current.get(field)
    if (first is None) != (second is None):
        structural_errors.append(f"frame_{frame_index}:public_pose.{field}:availability")
        return
    if first is None:
        return
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    if first_array.shape != second_array.shape:
        structural_errors.append(f"frame_{frame_index}:public_pose.{field}:shape")
        return
    errors.append(scale * float(np.linalg.norm(first_array - second_array)))


def _append_scalar_error(
    reference: Mapping[str, Any],
    current: Mapping[str, Any],
    field: str,
    *,
    frame_index: int,
    structural_errors: list[str],
    errors: list[float],
    difference: Any,
) -> None:
    first = reference.get(field)
    second = current.get(field)
    if (first is None) != (second is None):
        structural_errors.append(f"frame_{frame_index}:public_pose.{field}:availability")
        return
    if first is not None:
        errors.append(float(difference(float(first), float(second))))


def _compare_public_pose(
    reference: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    frame_index: int,
    structural_errors: list[str],
    public_center_errors_mm: list[float],
    public_axis_errors: list[float],
    public_angle_errors_deg: list[float],
    feasible_errors_mm: list[float],
    confidence_errors: list[float],
) -> None:
    reference_keys = set(reference)
    current_keys = set(current)
    if reference_keys != current_keys:
        structural_errors.append(f"frame_{frame_index}:public_pose:key_set")
    exact_fields = set(_PUBLIC_EXACT_FIELDS)
    for field in exact_fields:
        if reference.get(field) != current.get(field):
            structural_errors.append(f"frame_{frame_index}:public_pose.{field}")

    reference_confidence = reference["confidence"]
    current_confidence = current["confidence"]
    reference_confidence_keys = set(reference_confidence)
    current_confidence_keys = set(current_confidence)
    if reference_confidence_keys != current_confidence_keys:
        structural_errors.append(
            f"frame_{frame_index}:public_pose.confidence:key_set"
        )
    confidence_exact_fields = {
        "geometry_valid",
        "full_pose_valid",
        "absolute_base_pose_valid",
        "reasons",
    }
    for field in confidence_exact_fields:
        if reference_confidence.get(field) != current_confidence.get(field):
            structural_errors.append(
                f"frame_{frame_index}:public_pose.confidence.{field}"
            )
    for field in sorted(
        (reference_confidence_keys & current_confidence_keys)
        - confidence_exact_fields
        - {"per_field"}
    ):
        if reference_confidence[field] != current_confidence[field]:
            structural_errors.append(
                f"frame_{frame_index}:public_pose.confidence.{field}"
            )
    reference_per_field = reference_confidence["per_field"]
    current_per_field = current_confidence["per_field"]
    if set(reference_per_field) != set(current_per_field):
        structural_errors.append(
            f"frame_{frame_index}:public_pose.confidence.per_field:key_set"
        )
    for field in reference_per_field.keys() & current_per_field.keys():
        confidence_errors.append(
            abs(float(reference_per_field[field]) - float(current_per_field[field]))
        )

    for field in _PUBLIC_CENTER_FIELDS:
        _append_vector_error(
            reference,
            current,
            field,
            frame_index=frame_index,
            scale=1000.0,
            structural_errors=structural_errors,
            errors=public_center_errors_mm,
        )
    for field in _PUBLIC_AXIS_FIELDS:
        _append_vector_error(
            reference,
            current,
            field,
            frame_index=frame_index,
            scale=1.0,
            structural_errors=structural_errors,
            errors=public_axis_errors,
        )
    for field in _PUBLIC_LINE_RAD_FIELDS:
        _append_scalar_error(
            reference,
            current,
            field,
            frame_index=frame_index,
            structural_errors=structural_errors,
            errors=public_angle_errors_deg,
            difference=lambda first, second: math.degrees(
                abs((first - second + math.pi / 2.0) % math.pi - math.pi / 2.0)
            ),
        )
    for field in _PUBLIC_LINE_DEG_FIELDS:
        _append_scalar_error(
            reference,
            current,
            field,
            frame_index=frame_index,
            structural_errors=structural_errors,
            errors=public_angle_errors_deg,
            difference=_line_angle_error_deg,
        )
    for field in _PUBLIC_SCALAR_DEG_FIELDS:
        _append_scalar_error(
            reference,
            current,
            field,
            frame_index=frame_index,
            structural_errors=structural_errors,
            errors=public_angle_errors_deg,
            difference=lambda first, second: abs(first - second),
        )

    reference_feasible = reference.get("center_feasible_set")
    current_feasible = current.get("center_feasible_set")
    if (reference_feasible is None) != (current_feasible is None):
        structural_errors.append(
            f"frame_{frame_index}:public_pose.center_feasible_set:availability"
        )
    elif reference_feasible is not None:
        if set(reference_feasible) != set(current_feasible):
            structural_errors.append(
                f"frame_{frame_index}:public_pose.center_feasible_set:key_set"
            )
        for axis in reference_feasible.keys() & current_feasible.keys():
            first = np.asarray(reference_feasible[axis], dtype=np.float64)
            second = np.asarray(current_feasible[axis], dtype=np.float64)
            if first.shape != second.shape:
                structural_errors.append(
                    f"frame_{frame_index}:public_pose.center_feasible_set.{axis}:shape"
                )
            else:
                feasible_errors_mm.extend(
                    (1000.0 * np.abs(first - second)).tolist()
                )

    known_fields = (
        exact_fields
        | set(_PUBLIC_CENTER_FIELDS)
        | set(_PUBLIC_AXIS_FIELDS)
        | set(_PUBLIC_LINE_RAD_FIELDS)
        | set(_PUBLIC_LINE_DEG_FIELDS)
        | set(_PUBLIC_SCALAR_DEG_FIELDS)
        | {"center_feasible_set", "confidence"}
    )
    for field in sorted((reference_keys & current_keys) - known_fields):
        if reference[field] != current[field]:
            structural_errors.append(f"frame_{frame_index}:public_pose.{field}")


def compare_reports(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    max_p50_ratio: float = 0.92,
    max_p95_ratio: float = 0.90,
    min_throughput_ratio: float = 1.08,
    max_center_p95_mm: float = 0.10,
    max_center_mm: float = 0.50,
    max_yaw_p95_deg: float = 0.05,
    max_yaw_deg: float = 0.20,
    max_confidence_delta: float = 0.01,
    max_axis_vector_delta: float = 1e-6,
) -> dict[str, Any]:
    """Compare a candidate against a same-host baseline and return a gate."""

    validation_errors = _report_validation_errors(
        baseline, label="baseline"
    ) + _report_validation_errors(candidate, label="candidate")
    if validation_errors:
        return {
            "status": "inconclusive",
            "comparable": False,
            "behavior_pass": False,
            "speed_pass": False,
            "report_validation_errors": validation_errors,
        }

    fixture_matches = all(
        baseline["fixture"][key] == candidate["fixture"][key]
        for key in _REQUIRED_FIXTURE_HASH_KEYS
    )
    settings_matches = baseline.get("settings") == candidate.get("settings")
    environment_matches = (
        baseline.get("environment_fingerprint")
        == candidate.get("environment_fingerprint")
    )
    baseline_frames = baseline["behavior"]["frames"]
    candidate_frames = candidate["behavior"]["frames"]
    structural_errors: list[str] = []
    center_errors_mm: list[float] = []
    top_center_errors_mm: list[float] = []
    yaw_errors_deg: list[float] = []
    public_center_errors_mm: list[float] = []
    public_axis_errors: list[float] = []
    public_angle_errors_deg: list[float] = []
    feasible_errors_mm: list[float] = []
    confidence_errors: list[float] = []
    if len(baseline_frames) != len(candidate_frames):
        structural_errors.append("frame_count_changed")
    for reference, current in zip(baseline_frames, candidate_frames, strict=False):
        index = int(reference["frame_index"])
        reference_keys = set(reference)
        current_keys = set(current)
        if reference_keys != current_keys:
            structural_errors.append(f"frame_{index}:key_set")
        exact_frame_fields = {
            "frame_index",
            "frame_id",
            "timestamp_ms",
            "full_pose_valid",
            "absolute_valid",
            "base_registration",
            "canonical_reference_deg",
            "observability",
            "reasons",
        }
        for key in exact_frame_fields:
            if reference.get(key) != current.get(key):
                structural_errors.append(f"frame_{index}:{key}")
        confidence_errors.append(
            abs(
                float(reference["yaw_confidence"])
                - float(current["yaw_confidence"])
            )
        )
        _compare_public_pose(
            reference["public_pose"],
            current["public_pose"],
            frame_index=index,
            structural_errors=structural_errors,
            public_center_errors_mm=public_center_errors_mm,
            public_axis_errors=public_axis_errors,
            public_angle_errors_deg=public_angle_errors_deg,
            feasible_errors_mm=feasible_errors_mm,
            confidence_errors=confidence_errors,
        )
        reference_center = reference.get("box_center_base_xyz_m")
        current_center = current.get("box_center_base_xyz_m")
        reference_top = reference.get("top_center_base_xyz_m")
        current_top = current.get("top_center_base_xyz_m")
        reference_yaw = reference.get("yaw_base_signed_deg")
        current_yaw = current.get("yaw_base_signed_deg")
        if (reference_center is None) != (current_center is None):
            structural_errors.append(f"frame_{index}:box_center_availability")
        elif reference_center is not None:
            center_errors_mm.append(
                1000.0
                * float(
                    np.linalg.norm(
                        np.asarray(reference_center) - np.asarray(current_center)
                    )
                )
            )
        if (reference_top is None) != (current_top is None):
            structural_errors.append(f"frame_{index}:top_center_availability")
        elif reference_top is not None:
            top_center_errors_mm.append(
                1000.0
                * float(
                    np.linalg.norm(np.asarray(reference_top) - np.asarray(current_top))
                )
            )
        if (reference_yaw is None) != (current_yaw is None):
            structural_errors.append(f"frame_{index}:yaw_availability")
        elif reference_yaw is not None:
            yaw_errors_deg.append(_line_angle_error_deg(reference_yaw, current_yaw))
        known_frame_fields = exact_frame_fields | {
            "yaw_confidence",
            "public_pose",
            "box_center_base_xyz_m",
            "top_center_base_xyz_m",
            "yaw_base_signed_deg",
        }
        for key in sorted((reference_keys & current_keys) - known_frame_fields):
            if reference[key] != current[key]:
                structural_errors.append(f"frame_{index}:{key}")

    center = _numeric_error_summary(center_errors_mm)
    top_center = _numeric_error_summary(top_center_errors_mm)
    yaw = _numeric_error_summary(yaw_errors_deg)
    public_center = _numeric_error_summary(public_center_errors_mm)
    public_axis = _numeric_error_summary(public_axis_errors)
    public_angle = _numeric_error_summary(public_angle_errors_deg)
    feasible = _numeric_error_summary(feasible_errors_mm)
    confidence = _numeric_error_summary(confidence_errors)
    reference_latency = baseline["latency"]
    current_latency = candidate["latency"]
    reference_frame = reference_latency["per_frame_median_ms"]
    current_frame = current_latency["per_frame_median_ms"]
    reference_throughput = float(
        reference_latency["per_pass"]["median_throughput_fps"]
    )
    current_throughput = float(
        current_latency["per_pass"]["median_throughput_fps"]
    )
    p50_ratio = float(current_frame["p50"]) / float(reference_frame["p50"])
    p95_ratio = float(current_frame["p95"]) / float(reference_frame["p95"])
    throughput_ratio = current_throughput / reference_throughput

    baseline_stable = bool(baseline["stability"]["outputs_identical"]) and bool(
        baseline["stability"]["p95_mad_ratio_within_3_percent"]
    )
    candidate_stable = bool(candidate["stability"]["outputs_identical"]) and bool(
        candidate["stability"]["p95_mad_ratio_within_3_percent"]
    )
    comparable = (
        fixture_matches
        and settings_matches
        and environment_matches
        and baseline_stable
        and candidate_stable
    )
    behavior_pass = (
        not structural_errors
        and center["p95"] <= max_center_p95_mm
        and center["maximum"] <= max_center_mm
        and top_center["p95"] <= max_center_p95_mm
        and top_center["maximum"] <= max_center_mm
        and yaw["p95"] <= max_yaw_p95_deg
        and yaw["maximum"] <= max_yaw_deg
        and public_center["p95"] <= max_center_p95_mm
        and public_center["maximum"] <= max_center_mm
        and feasible["p95"] <= max_center_p95_mm
        and feasible["maximum"] <= max_center_mm
        and public_angle["p95"] <= max_yaw_p95_deg
        and public_angle["maximum"] <= max_yaw_deg
        and public_axis["maximum"] <= max_axis_vector_delta
        and confidence["maximum"] <= max_confidence_delta
    )
    speed_pass = (
        p50_ratio <= max_p50_ratio
        and p95_ratio <= max_p95_ratio
        and throughput_ratio >= min_throughput_ratio
    )
    status = (
        "pass"
        if comparable and behavior_pass and speed_pass
        else "inconclusive"
        if not comparable
        else "fail"
    )
    structural_errors = list(dict.fromkeys(structural_errors))
    return to_jsonable(
        {
            "status": status,
            "comparable": comparable,
            "fixture_matches": fixture_matches,
            "settings_matches": settings_matches,
            "environment_matches": environment_matches,
            "baseline_stable": baseline_stable,
            "candidate_stable": candidate_stable,
            "behavior_pass": behavior_pass,
            "speed_pass": speed_pass,
            "structural_errors": structural_errors,
            "numeric_equivalence": {
                "box_center_error_mm": center,
                "top_center_error_mm": top_center,
                "yaw_line_error_deg": yaw,
                "public_center_error_mm": public_center,
                "public_axis_vector_error": public_axis,
                "public_angle_error_deg": public_angle,
                "feasible_endpoint_error_mm": feasible,
                "per_field_confidence_absolute_error": confidence,
            },
            "speed": {
                "p50_ratio": p50_ratio,
                "p95_ratio": p95_ratio,
                "throughput_ratio": throughput_ratio,
                "baseline": {
                    "p50_ms": reference_frame["p50"],
                    "p95_ms": reference_frame["p95"],
                    "throughput_fps": reference_throughput,
                },
                "candidate": {
                    "p50_ms": current_frame["p50"],
                    "p95_ms": current_frame["p95"],
                    "throughput_fps": current_throughput,
                },
                "thresholds": {
                    "max_p50_ratio": max_p50_ratio,
                    "max_p95_ratio": max_p95_ratio,
                    "min_throughput_ratio": min_throughput_ratio,
                },
            },
        }
    )


def _write_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists; pass --overwrite: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                to_jsonable(payload),
                stream,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
        Path(temporary_name).replace(path)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _validate_output_path(output: Path, protected_inputs: Sequence[Path]) -> None:
    if output.suffix.lower() != ".json":
        raise ValueError("benchmark output must use a .json suffix")
    resolved_output = output.expanduser().resolve(strict=False)
    for source in protected_inputs:
        resolved_source = source.expanduser().resolve(strict=False)
        if resolved_output == resolved_source:
            raise ValueError(
                "benchmark output must be distinct from every input: "
                f"{output} aliases {source}"
            )


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--warmup-passes", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    capture = subparsers.add_parser("capture", help="write an estimator baseline")
    _common_arguments(capture)
    compare = subparsers.add_parser("compare", help="compare against a baseline")
    _common_arguments(compare)
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--max-p50-ratio", type=float, default=0.92)
    compare.add_argument("--max-p95-ratio", type=float, default=0.90)
    compare.add_argument("--min-throughput-ratio", type=float, default=1.08)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protected_inputs = [
        args.session / MANIFEST_NAME,
        args.calibration,
        args.config,
    ]
    baseline: Mapping[str, Any] | None = None
    if args.mode == "compare":
        protected_inputs.append(args.baseline)
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    _validate_output_path(args.output, protected_inputs)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"output already exists; pass --overwrite: {args.output}"
        )
    report = benchmark_session(
        args.session,
        args.calibration,
        args.config,
        warmup_passes=args.warmup_passes,
        repeats=args.repeats,
    )
    if args.mode == "capture":
        _write_json(args.output, report, overwrite=args.overwrite)
        print(json.dumps({"status": "captured", "output": str(args.output)}, indent=2))
        return 0
    if baseline is None:  # pragma: no cover - narrowed by argparse mode
        raise RuntimeError("compare mode requires a loaded baseline")
    comparison = compare_reports(
        baseline,
        report,
        max_p50_ratio=args.max_p50_ratio,
        max_p95_ratio=args.max_p95_ratio,
        min_throughput_ratio=args.min_throughput_ratio,
    )
    combined = {"benchmark": report, "comparison": comparison}
    _write_json(args.output, combined, overwrite=args.overwrite)
    print(json.dumps(comparison, allow_nan=False, indent=2, sort_keys=True))
    if comparison["status"] == "pass":
        return 0
    return 2 if comparison["status"] == "inconclusive" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["benchmark_session", "build_parser", "compare_reports", "main"]
