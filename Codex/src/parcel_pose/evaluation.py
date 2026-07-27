"""Recorded-session performance evaluation and annotated MP4 rendering.

The video is a perception diagnostic, not a robot command surface.  A base
pose derived from FK plus a nominal camera mount stays explicitly labelled
``nominal_unverified`` until an independent base-referenced calibration exists.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
import os
import platform
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .angles import classify_canonical_angle_deg, normalize_signed_line_angle_deg
from .calibration import factory_extrinsics_to_transform
from .estimator import EstimationEvidence, ParcelPoseEstimator
from .models import Calibration, CameraIntrinsics, EstimatorConfig, PoseEstimate
from .output import dumps_strict, to_jsonable
from .projection import unproject_plane_points
from .recording import MANIFEST_NAME, SessionReader
from .transforms import transform_points
from .visualization import project_points_to_pixels


FloatArray = NDArray[np.float64]
ImageArray = NDArray[np.uint8]
_REGISTRATION_CAVEATS = {"absolute_base_transform_unvalidated"}


def _cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to create an evaluation video") from exc
    return cv2


@dataclass(frozen=True, slots=True)
class BasePoseDiagnostic:
    """One metric box pose expressed in the RB-Y1 base frame."""

    box_center_xyz_m: tuple[float, float, float]
    top_center_xyz_m: tuple[float, float, float]
    yaw_mod_180_deg: float
    yaw_signed_deg: float
    canonical_reference_deg: int | None
    canonical_residual_deg: float | None
    registration: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "box_center_xyz_m": list(self.box_center_xyz_m),
            "top_center_xyz_m": list(self.top_center_xyz_m),
            "long_axis_yaw_base_deg": self.yaw_mod_180_deg,
            "long_axis_yaw_base_signed_deg": self.yaw_signed_deg,
            "canonical_reference_deg": self.canonical_reference_deg,
            "canonical_residual_deg": self.canonical_residual_deg,
            "registration": self.registration,
        }


def _table_normal_base(calibration: Calibration) -> FloatArray:
    if calibration.table_plane is None:
        raise ValueError("table plane is required for a box-volume center")
    plane = calibration.table_plane
    if plane.frame in {"base", calibration.base_frame}:
        normal = np.asarray(plane.normal, dtype=np.float64)
    elif plane.frame in {"depth", calibration.depth_frame}:
        transform = calibration.T_base_from_depth
        if transform is None:
            raise ValueError("a complete T_base_from_depth chain is required")
        normal = transform[:3, :3] @ plane.normal
    else:
        raise ValueError(f"unsupported table-plane frame for base output: {plane.frame}")
    return normal / max(float(np.linalg.norm(normal)), 1e-12)


def base_pose_from_estimate(
    estimate: PoseEstimate,
    calibration: Calibration,
) -> BasePoseDiagnostic | None:
    """Recover a clearly gated base-frame pose for display and evaluation.

    ``estimate.center_depth_m`` is the fitted top-surface center.  The requested
    physical box center is half the known 150 mm height below it along the
    calibrated table normal.
    """

    transform = calibration.T_base_from_depth
    if transform is None or not estimate.full_pose_valid or estimate.center_depth_m is None:
        return None
    top_center = np.asarray(transform_points(estimate.center_depth_m, transform), dtype=np.float64)
    box_center = top_center - 0.5 * float(estimate.box_model.height_m) * _table_normal_base(
        calibration
    )

    yaw_rad: float | None = None
    nominal = estimate.diagnostics.get("nominal_unverified_base")
    if isinstance(nominal, Mapping) and nominal.get("yaw_rad") is not None:
        yaw_rad = float(nominal["yaw_rad"])
    elif estimate.frame == calibration.base_frame and estimate.yaw_rad is not None:
        yaw_rad = float(estimate.yaw_rad)
    elif estimate.long_axis_base_xy is not None:
        yaw_rad = math.atan2(estimate.long_axis_base_xy[1], estimate.long_axis_base_xy[0])
    if yaw_rad is None or not math.isfinite(yaw_rad):
        return None

    yaw_deg = math.degrees(yaw_rad) % 180.0
    canonical = classify_canonical_angle_deg(
        yaw_deg,
        uncertainty_deg=0.0,
        long_short_assignment_valid=estimate.observability.get("yaw") != "underconstrained",
    )
    registration = "validated" if calibration.absolute_base_validated else "nominal_unverified"
    return BasePoseDiagnostic(
        box_center_xyz_m=tuple(float(value) for value in box_center),
        top_center_xyz_m=tuple(float(value) for value in top_center),
        yaw_mod_180_deg=yaw_deg,
        yaw_signed_deg=normalize_signed_line_angle_deg(yaw_deg),
        canonical_reference_deg=canonical.reference_deg,
        canonical_residual_deg=canonical.residual_deg,
        registration=registration,
    )


def _project_depth_points_to_color(
    points_depth_m: Any,
    color_from_depth: FloatArray,
    color_intrinsics: CameraIntrinsics,
) -> FloatArray:
    points_color = transform_points(points_depth_m, color_from_depth)
    return project_points_to_pixels(points_color, color_intrinsics)


def _draw_text_lines(image: ImageArray, lines: Sequence[str], *, warning: bool) -> None:
    cv2 = _cv2()
    panel_height = min(image.shape[0], 24 + 21 * len(lines))
    panel_width = min(image.shape[1], 610)
    panel = image[:panel_height, :panel_width].copy()
    panel[:] = (22, 22, 22)
    cv2.addWeighted(panel, 0.78, image[:panel_height, :panel_width], 0.22, 0.0, image[:panel_height, :panel_width])
    for index, line in enumerate(lines):
        color = (70, 190, 255) if warning and index == 1 else (245, 245, 245)
        y = 22 + 21 * index
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)


def draw_evaluation_overlay(
    image_bgr: Any,
    estimate: PoseEstimate,
    base_pose: BasePoseDiagnostic | None,
    *,
    evidence: EstimationEvidence | None,
    color_from_depth: FloatArray,
    color_intrinsics: CameraIntrinsics,
    frame_index: int,
    frame_count: int,
    estimator_latency_ms: float,
) -> ImageArray:
    """Render raw RGB with depth-derived metric evidence and base-pose text."""

    cv2 = _cv2()
    image = np.asarray(image_bgr)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("raw RGB image must be uint8 HxWx3 BGR")
    output = image.copy()
    height, width = output.shape[:2]

    if evidence is not None and evidence.projection.count:
        points = evidence.projection.points_3d_m
        step = max(1, len(points) // 2_500)
        pixels = _project_depth_points_to_color(points[::step], color_from_depth, color_intrinsics)
        finite = np.isfinite(pixels).all(axis=1)
        integer = np.rint(pixels[finite]).astype(np.int32)
        inside = (
            (integer[:, 0] >= 0)
            & (integer[:, 0] < width)
            & (integer[:, 1] >= 0)
            & (integer[:, 1] < height)
        )
        output[integer[inside, 1], integer[inside, 0]] = (70, 190, 70)

    if evidence is not None and evidence.rectangle.corners_xy_m is not None:
        corners_depth = unproject_plane_points(
            evidence.rectangle.corners_xy_m,
            evidence.projection.plane,
            origin=evidence.projection.origin_3d_m,
            basis=(evidence.projection.basis_u_3d, evidence.projection.basis_v_3d),
        )
        pixels = _project_depth_points_to_color(
            corners_depth, color_from_depth, color_intrinsics
        )
        if np.isfinite(pixels).all():
            polygon = np.rint(pixels).astype(np.int32).reshape(-1, 1, 2)
            color = (40, 220, 255) if estimate.full_pose_valid else (0, 128, 255)
            cv2.polylines(output, [polygon], True, color, 2, cv2.LINE_AA)

    if estimate.center_depth_m is not None:
        center_pixel = _project_depth_points_to_color(
            [estimate.center_depth_m], color_from_depth, color_intrinsics
        )[0]
        if np.isfinite(center_pixel).all():
            center_uv = tuple(np.rint(center_pixel).astype(int))
            if 0 <= center_uv[0] < width and 0 <= center_uv[1] < height:
                cv2.drawMarker(
                    output,
                    center_uv,
                    (40, 220, 255),
                    cv2.MARKER_CROSS,
                    16,
                    2,
                    cv2.LINE_AA,
                )

    if base_pose is None:
        lines = [
            f"BOX TRACK  ABSTAIN   frame {frame_index + 1}/{frame_count}",
            "base pose: unavailable for this frame",
            "reason: " + ", ".join(estimate.reasons[:2]),
            f"estimator latency: {estimator_latency_ms:.1f} ms",
            "NO GT: availability/latency test only",
        ]
        _draw_text_lines(output, lines, warning=True)
        return output

    center = base_pose.box_center_xyz_m
    reference = "--" if base_pose.canonical_reference_deg is None else str(base_pose.canonical_reference_deg)
    residual = "--" if base_pose.canonical_residual_deg is None else f"{base_pose.canonical_residual_deg:+.1f}"
    yaw_confidence = float(estimate.per_field_confidence.get("yaw", 0.0))
    lines = [
        f"BOX TRACK  VALID   frame {frame_index + 1}/{frame_count}",
        f"registration: {base_pose.registration} (RB-Y1 M v1.2 FK + nominal mount)",
        f"box center base [m]  x={center[0]:+.3f}  y={center[1]:+.3f}  z={center[2]:+.3f}",
        f"long-axis yaw base={base_pose.yaw_signed_deg:+.1f} deg  (mod180={base_pose.yaw_mod_180_deg:.1f})",
        f"canonical reference={reference} deg  residual={residual} deg",
        f"yaw confidence={yaw_confidence:.2f}  estimator latency={estimator_latency_ms:.1f} ms",
        "NO GT: availability/latency test only",
    ]
    _draw_text_lines(output, lines, warning=base_pose.registration != "validated")
    return output


def _timing_summary(timestamps_ms: Sequence[float], *, nominal_fps: float) -> dict[str, float]:
    values = np.asarray(timestamps_ms, dtype=np.float64)
    if len(values) >= 2:
        deltas = np.diff(values)
        positive = deltas[np.isfinite(deltas) & (deltas > 0.0)]
    else:
        positive = np.empty(0, dtype=np.float64)
    if len(positive):
        duration_ms = float(values[-1] - values[0])
        effective_fps = 1000.0 * float(len(values) - 1) / duration_ms
        median_interval_ms = float(np.median(positive))
        median_arrival_fps = 1000.0 / median_interval_ms
        p95_interval_ms = float(np.percentile(positive, 95))
        max_interval_ms = float(np.max(positive))
    else:
        effective_fps = float(nominal_fps)
        duration_ms = 1000.0 * max(0, len(values) - 1) / max(effective_fps, 1e-9)
        median_interval_ms = 1000.0 / max(effective_fps, 1e-9)
        median_arrival_fps = effective_fps
        p95_interval_ms = median_interval_ms
        max_interval_ms = median_interval_ms
    return {
        "nominal_stream_fps": float(nominal_fps),
        "recorded_duration_sec": duration_ms / 1000.0,
        "effective_stored_fps": effective_fps,
        "median_interval_ms": median_interval_ms,
        "median_arrival_fps": median_arrival_fps,
        "p95_interval_ms": p95_interval_ms,
        "max_interval_ms": max_interval_ms,
    }


def _latency_summary(values_ms: Sequence[float]) -> dict[str, float | None]:
    if not values_ms:
        return {"mean_ms": None, "p50_ms": None, "p95_ms": None, "max_ms": None}
    values = np.asarray(values_ms, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": float(np.max(values)),
    }


def _longest_run(values: Sequence[bool], target: bool) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if bool(value) is target else 0
        longest = max(longest, current)
    return longest


def _manifest_timestamps(session_root: Path) -> list[float]:
    try:
        payload = json.loads((session_root / MANIFEST_NAME).read_text(encoding="utf-8"))
        frames = payload["frames"]
        return [float(frame["depth_timestamp_ms"]) for frame in frames]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot derive recording timing from manifest: {exc}") from exc


def _validate_distinct_output_paths(paths: Sequence[Path]) -> None:
    """Reject aliases before any output is opened or replaced."""

    resolved: dict[Path, Path] = {}
    for path in paths:
        canonical = path.expanduser().resolve(strict=False)
        previous = resolved.get(canonical)
        if previous is not None:
            raise ValueError(
                "evaluation output paths must be distinct: "
                f"{previous} and {path} resolve to {canonical}"
            )
        resolved[canonical] = path


def _allocate_temporary_video(video_path: Path) -> Path:
    """Allocate an owned, collision-resistant temporary MP4 beside the target."""

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{video_path.stem}.",
        suffix=".mp4",
        dir=video_path.parent,
    )
    os.close(descriptor)
    return Path(temporary_name)


def evaluate_session_video(
    session_root: str | Path,
    calibration: Calibration,
    estimator_config: EstimatorConfig,
    output_mp4: str | Path,
    *,
    output_summary: str | Path | None = None,
    output_jsonl: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Process every recorded frame, write an MP4, and return measured metrics."""

    if calibration.T_base_from_depth is None:
        raise ValueError("evaluation video requires a complete T_base_from_depth chain")
    session_path = Path(session_root)
    video_path = Path(output_mp4)
    summary_path = None if output_summary is None else Path(output_summary)
    jsonl_path = None if output_jsonl is None else Path(output_jsonl)
    targets = [path for path in (video_path, summary_path, jsonl_path) if path is not None]
    _validate_distinct_output_paths(targets)
    existing = [str(path) for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("output already exists; pass --overwrite: " + ", ".join(existing))

    reader = SessionReader(session_path)
    frame_count = len(reader)
    if frame_count == 0:
        raise ValueError("evaluation session contains no frames")
    timestamps = _manifest_timestamps(session_path)
    if len(timestamps) != frame_count:
        raise ValueError("manifest timestamp count does not match frame count")
    timing = _timing_summary(
        timestamps,
        nominal_fps=float(reader.metadata.depth_profile.intrinsics.fps),
    )
    output_fps = float(timing["effective_stored_fps"])
    color_intrinsics = reader.metadata.color_profile.intrinsics
    color_from_depth = factory_extrinsics_to_transform(reader.metadata.depth_to_color)
    estimator = ParcelPoseEstimator(
        reader.metadata.depth_profile.intrinsics,
        calibration,
        estimator_config,
    )

    cv2 = _cv2()
    video_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_video = _allocate_temporary_video(video_path)
    writer = cv2.VideoWriter(
        str(temporary_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (color_intrinsics.width, color_intrinsics.height),
    )
    if not writer.isOpened():
        writer.release()
        temporary_video.unlink(missing_ok=True)
        raise RuntimeError("OpenCV could not open an MP4V video writer")

    estimator_latencies: list[float] = []
    render_latencies: list[float] = []
    full_valid: list[bool] = []
    base_available: list[bool] = []
    records: list[dict[str, Any]] = []
    base_centers: list[tuple[float, float, float]] = []
    base_yaws: list[float] = []
    abstention_reasons: Counter[str] = Counter()
    wall_start = time.perf_counter()
    try:
        for frame_index, frame in enumerate(reader):
            estimate_start = time.perf_counter()
            estimate = estimator.estimate(
                frame.raw_depth_z16,
                depth_scale=reader.metadata.depth_scale_m,
                timestamp_ms=frame.depth_timestamp_ms,
                frame_id=frame.depth_frame_number,
            )
            estimator_ms = 1000.0 * (time.perf_counter() - estimate_start)
            estimator_latencies.append(estimator_ms)
            pose = base_pose_from_estimate(estimate, calibration)
            is_full_valid = bool(estimate.full_pose_valid)
            is_base_available = pose is not None
            full_valid.append(is_full_valid)
            base_available.append(is_base_available)
            if not is_full_valid:
                causes = [
                    reason
                    for reason in estimate.reasons
                    if reason not in _REGISTRATION_CAVEATS
                ]
                abstention_reasons.update(causes or ("unspecified",))
            if pose is not None:
                base_centers.append(pose.box_center_xyz_m)
                base_yaws.append(pose.yaw_signed_deg)

            render_start = time.perf_counter()
            overlay = draw_evaluation_overlay(
                frame.raw_color_bgr,
                estimate,
                pose,
                evidence=estimator.last_evidence,
                color_from_depth=color_from_depth,
                color_intrinsics=color_intrinsics,
                frame_index=frame_index,
                frame_count=frame_count,
                estimator_latency_ms=estimator_ms,
            )
            writer.write(overlay)
            render_ms = 1000.0 * (time.perf_counter() - render_start)
            render_latencies.append(render_ms)
            records.append(
                {
                    "frame_index": frame_index,
                    "frame_id": estimate.frame_id,
                    "timestamp_ms": estimate.timestamp_ms,
                    "full_pose_valid": is_full_valid,
                    "base_pose": None if pose is None else pose.to_dict(),
                    "yaw_confidence": float(estimate.per_field_confidence.get("yaw", 0.0)),
                    "estimator_latency_ms": estimator_ms,
                    "reasons": list(estimate.reasons),
                }
            )
    except BaseException:
        writer.release()
        temporary_video.unlink(missing_ok=True)
        raise
    writer.release()
    wall_seconds = time.perf_counter() - wall_start

    try:
        capture = cv2.VideoCapture(str(temporary_video))
        try:
            if not capture.isOpened():
                raise RuntimeError("generated MP4 cannot be reopened")
            encoded_frames = int(round(float(capture.get(cv2.CAP_PROP_FRAME_COUNT))))
            encoded_fps = float(capture.get(cv2.CAP_PROP_FPS))
            ok, first_frame = capture.read()
            if not ok or first_frame.shape[:2] != (
                color_intrinsics.height,
                color_intrinsics.width,
            ):
                raise RuntimeError("generated MP4 failed frame/decode verification")
        finally:
            capture.release()
        if abs(encoded_frames - frame_count) > 1:
            raise RuntimeError(
                "generated MP4 frame count mismatch: "
                f"expected {frame_count}, got {encoded_frames}"
            )
        temporary_video.replace(video_path)
    except BaseException:
        temporary_video.unlink(missing_ok=True)
        raise

    valid_count = sum(full_valid)
    base_count = sum(base_available)
    registration = "validated" if calibration.absolute_base_validated else "nominal_unverified"
    table_normal_base = _table_normal_base(calibration)
    center_range: dict[str, Any] | None = None
    if base_centers:
        centers = np.asarray(base_centers, dtype=np.float64)
        center_range = {
            "minimum_xyz_m": np.min(centers, axis=0).tolist(),
            "maximum_xyz_m": np.max(centers, axis=0).tolist(),
        }
    yaw_range: dict[str, float] | None = None
    if base_yaws:
        yaw_range = {
            "minimum_signed_deg": float(np.min(base_yaws)),
            "maximum_signed_deg": float(np.max(base_yaws)),
        }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "session": str(session_path),
        "video": {
            "path": str(video_path),
            "codec": "mp4v",
            "width": color_intrinsics.width,
            "height": color_intrinsics.height,
            "output_fps": output_fps,
            "verified_encoded_fps": encoded_fps,
            "verified_frame_count": encoded_frames,
        },
        "input_timing": timing,
        "pose_availability": {
            "frames": frame_count,
            "full_pose_valid_frames": valid_count,
            "full_pose_valid_ratio": valid_count / frame_count,
            "base_pose_available_frames": base_count,
            "base_pose_available_ratio": base_count / frame_count,
            "absolute_base_pose_frames": base_count if calibration.absolute_base_validated else 0,
            "nominal_unverified_base_pose_frames": 0 if calibration.absolute_base_validated else base_count,
            "abstained_frames": frame_count - valid_count,
            "longest_valid_run_frames": _longest_run(full_valid, True),
            "longest_abstention_run_frames": _longest_run(full_valid, False),
            "abstention_reason_counts": dict(sorted(abstention_reasons.items())),
        },
        "processing": {
            "estimator_latency": _latency_summary(estimator_latencies),
            "render_encode_latency": _latency_summary(render_latencies),
            "total_wall_sec": wall_seconds,
            "end_to_end_fps": frame_count / max(wall_seconds, 1e-12),
        },
        "runtime_environment": {
            "timing_scope": "offline evaluation on this host, not target-robot runtime",
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "opencv_version": str(cv2.__version__),
            "numpy_version": str(np.__version__),
        },
        "base_registration": {
            "status": registration,
            "calibration_state": calibration.state.value,
            "independently_validated": calibration.absolute_base_validated,
            "input": calibration.diagnostics.get("base_registration_input"),
            "notes": list(calibration.notes),
        },
        "calibration_sanity": {
            "table_normal_base": table_normal_base.tolist(),
            "table_normal_tilt_from_base_z_deg": math.degrees(
                math.acos(float(np.clip(table_normal_base[2], -1.0, 1.0)))
            ),
        },
        "tracked_box_center_range": center_range,
        "tracked_yaw_range": yaw_range,
        "accuracy_evaluation": {
            "available": False,
            "reason": "box_complex has no independent ground-truth center/yaw labels",
        },
    }
    summary = to_jsonable(summary)
    if jsonl_path is not None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_path.write_text(
            "\n".join(dumps_strict(record) for record in records) + "\n",
            encoding="utf-8",
        )
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(dumps_strict(summary, indent=2) + "\n", encoding="utf-8")
    return summary


__all__ = [
    "BasePoseDiagnostic",
    "base_pose_from_estimate",
    "draw_evaluation_overlay",
    "evaluate_session_video",
]
