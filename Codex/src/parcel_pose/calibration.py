"""Calibration artifact loading and empty-table plane estimation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from .models import Calibration, CalibrationState, Plane
from .plane import PlaneFitResult, fit_plane_ransac, signed_distances
from .recording import SessionReader
from .session import FactoryExtrinsics, SessionValidationError
from .transforms import make_transform, transform_from_euler_zyx


def factory_extrinsics_to_transform(extrinsics: FactoryExtrinsics) -> NDArray[np.float64]:
    # librealsense exposes rs2_extrinsics.rotation as a flat column-major
    # matrix.  Preserve that SDK storage contract in recordings and make the
    # conversion explicit here instead of relying on NumPy's row-major default.
    order = "F" if extrinsics.rotation_storage == "column_major" else "C"
    rotation = np.asarray(extrinsics.rotation, dtype=np.float64).reshape(3, 3, order=order)
    return make_transform(rotation, np.asarray(extrinsics.translation_m, dtype=np.float64))


def load_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON configuration {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON configuration must contain an object: {path}")
    return value


def nominal_calibration_from_config(
    config: Mapping[str, Any],
    *,
    E_color_from_depth: Any | None = None,
    T_base_from_head: Any | None = None,
    table_plane: Plane | None = None,
    state: CalibrationState = CalibrationState.NOMINAL,
    diagnostics: Mapping[str, Any] | None = None,
) -> Calibration:
    try:
        calibration_config = config["calibration"]
        frames = config["frames"]
        nominal = calibration_config["nominal_T_head_from_color"]
        translation = nominal["translation_m"]
        roll, pitch, yaw = nominal["euler_zyx_deg"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"nominal configuration is missing transform fields: {exc}") from exc
    formula = nominal.get("rotation_formula", "")
    if formula != "Rz(yaw) @ Ry(pitch) @ Rx(roll)":
        raise ValueError("unsupported Euler formula; expected Rz(yaw) @ Ry(pitch) @ Rx(roll)")
    return Calibration(
        state=state,
        table_plane=table_plane,
        T_base_from_head=T_base_from_head,
        T_head_from_color=transform_from_euler_zyx(
            translation, roll, pitch, yaw, degrees=True
        ),
        E_color_from_depth=E_color_from_depth,
        base_frame=str(frames.get("base", "base")),
        head_frame=str(frames.get("head_candidate", "link_head_2")),
        color_frame=str(frames.get("color", "d435_color_optical_frame")),
        depth_frame=str(frames.get("depth", "d435_depth_optical_frame")),
        notes=("nominal RGB-centered transform seed; base-plane XY/yaw unvalidated",),
        diagnostics={} if diagnostics is None else dict(diagnostics),
    )


def load_calibration(path: str | Path) -> Calibration:
    value = load_json(path)
    if "calibration" in value and "frames" in value:
        return nominal_calibration_from_config(value)
    return Calibration.from_dict(value)


def save_calibration(path: str | Path, calibration: Calibration) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(calibration.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return target


def _deproject_depth_sample(
    depth_z16: NDArray[np.uint16],
    *,
    scale_m: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    stride: int,
    min_depth_m: float,
    max_depth_m: float,
    roi_uv: tuple[int, int, int, int] | None = None,
) -> NDArray[np.float64]:
    height, width = depth_z16.shape
    if roi_uv is None:
        u0, v0, u1, v1 = 0, 0, width, height
    else:
        u0, v0, u1, v1 = (int(value) for value in roi_uv)
        if not (0 <= u0 < u1 <= width and 0 <= v0 < v1 <= height):
            raise ValueError(
                f"table ROI {(u0, v0, u1, v1)} is outside depth image {width}x{height}"
            )
    sampled = depth_z16[v0:v1:stride, u0:u1:stride].astype(np.float64) * scale_m
    v, u = np.indices(sampled.shape, dtype=np.float64)
    u = u0 + u * stride
    v = v0 + v * stride
    valid = np.isfinite(sampled) & (sampled >= min_depth_m) & (sampled <= max_depth_m)
    z = sampled[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    return np.column_stack((x, y, z))


def _even_sample(points: NDArray[np.float64], limit: int) -> NDArray[np.float64]:
    if points.shape[0] <= limit:
        return points
    indices = np.linspace(0, points.shape[0] - 1, limit, dtype=np.int64)
    return points[indices]


def fit_empty_table_plane_result(
    points_depth: Any,
    *,
    tolerance_m: float = 0.006,
    iterations: int = 300,
    min_inlier_ratio: float = 0.35,
    seed: int = 1729,
) -> PlaneFitResult:
    """Robustly fit an empty-table plane and retain acceptance evidence."""

    points = np.asarray(points_depth, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 100:
        raise ValueError("empty-table calibration needs at least 100 finite 3-D points")
    return fit_plane_ransac(
        points,
        tolerance_m=tolerance_m,
        iterations=iterations,
        min_inliers=100,
        min_inlier_ratio=min_inlier_ratio,
        seed=seed,
        frame="d435_depth_optical_frame",
        camera_origin=(0.0, 0.0, 0.0),
    )


def fit_empty_table_plane(
    points_depth: Any,
    *,
    trim_residual_m: float = 0.006,
    iterations: int = 300,
) -> Plane:
    """Compatibility wrapper returning the robust plane only."""

    return fit_empty_table_plane_result(
        points_depth,
        tolerance_m=trim_residual_m,
        iterations=iterations,
    ).plane


def _plane_fit_diagnostics(
    result: PlaneFitResult,
    frame_samples: list[tuple[int, NDArray[np.float64]]],
    *,
    tolerance_m: float,
    stride: int,
    roi_uv: tuple[int, int, int, int] | None,
    acceptance: Mapping[str, float],
) -> dict[str, Any]:
    residuals = np.asarray(result.residuals_m, dtype=np.float64)
    inlier_residuals = residuals[result.inlier_mask]
    per_frame: list[dict[str, Any]] = []
    for frame_id, points in frame_samples:
        frame_residuals = np.abs(signed_distances(points, result.plane))
        inliers = frame_residuals <= tolerance_m
        inlier_values = frame_residuals[inliers]
        per_frame.append(
            {
                "frame_id": frame_id,
                "sample_count": int(points.shape[0]),
                "inlier_ratio": float(np.mean(inliers)) if len(inliers) else 0.0,
                "median_residual_m": float(np.median(frame_residuals)),
                "p95_residual_m": float(np.percentile(frame_residuals, 95)),
                "inlier_rms_m": (
                    float(np.sqrt(np.mean(np.square(inlier_values))))
                    if len(inlier_values)
                    else None
                ),
            }
        )

    min_frame_ratio = min(item["inlier_ratio"] for item in per_frame)
    max_frame_p95 = max(item["p95_residual_m"] for item in per_frame)
    quality_checks = {
        "global_inlier_ratio": result.inlier_ratio
        >= float(acceptance["min_global_inlier_ratio"]),
        "global_inlier_rms_m": result.inlier_rms_m
        <= float(acceptance["max_global_inlier_rms_m"]),
        "minimum_frame_inlier_ratio": min_frame_ratio
        >= float(acceptance["min_frame_inlier_ratio"]),
        "maximum_frame_p95_residual_m": max_frame_p95
        <= float(acceptance["max_frame_p95_residual_m"]),
    }
    return {
        **result.to_dict(),
        "residual_m": {
            "median_all": float(np.median(residuals)),
            "p95_all": float(np.percentile(residuals, 95)),
            "median_inlier": float(np.median(inlier_residuals)),
            "p95_inlier": float(np.percentile(inlier_residuals, 95)),
            "max_inlier": float(np.max(inlier_residuals)),
        },
        "normal_faces_camera": bool(
            float(result.plane.normal @ -result.plane.point_on_plane()) > 0.0
        ),
        "sampling": {
            "stride": stride,
            "roi_uv": None if roi_uv is None else list(roi_uv),
        },
        "per_frame": per_frame,
        "acceptance_thresholds": dict(acceptance),
        "quality_checks": quality_checks,
        "quality_passed": all(quality_checks.values()),
    }


def calibrate_table_plane_from_session(
    session_root: str | Path,
    nominal_config: Mapping[str, Any],
    *,
    stride: int = 4,
    min_depth_m: float = 0.2,
    max_depth_m: float = 2.0,
    roi_uv: tuple[int, int, int, int] | None = None,
) -> Calibration:
    reader = SessionReader(session_root)
    intrinsics = reader.metadata.depth_profile.intrinsics
    options = dict(nominal_config.get("table_calibration", {}))
    if roi_uv is None and options.get("roi_uv") is not None:
        roi_uv = tuple(int(value) for value in options["roi_uv"])
    tolerance_m = float(options.get("ransac_tolerance_m", 0.006))
    iterations = int(options.get("ransac_iterations", 300))
    min_inlier_ratio = float(options.get("ransac_min_inlier_ratio", 0.35))
    max_fit_points = int(options.get("max_fit_points", 100_000))
    seed = int(options.get("random_seed", 1729))
    if stride <= 0 or max_fit_points < 100:
        raise ValueError("calibration stride must be positive and max_fit_points >= 100")
    per_frame_limit = max(100, int(math.ceil(max_fit_points / max(1, len(reader)))))
    frame_samples: list[tuple[int, NDArray[np.float64]]] = []
    for frame in reader:
        points = _deproject_depth_sample(
            frame.raw_depth_z16,
            scale_m=reader.metadata.depth_scale_m,
            fx=intrinsics.fx,
            fy=intrinsics.fy,
            cx=intrinsics.cx,
            cy=intrinsics.cy,
            stride=stride,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
            roi_uv=roi_uv,
        )
        if len(points):
            frame_samples.append(
                (frame.depth_frame_number, _even_sample(points, per_frame_limit))
            )
    if not frame_samples:
        raise SessionValidationError("empty-table calibration session has no frames")
    fit_points = _even_sample(
        np.concatenate([points for _, points in frame_samples], axis=0),
        max_fit_points,
    )
    fit_result = fit_empty_table_plane_result(
        fit_points,
        tolerance_m=tolerance_m,
        iterations=iterations,
        min_inlier_ratio=min_inlier_ratio,
        seed=seed,
    )
    acceptance = {
        "min_global_inlier_ratio": float(
            options.get("accept_min_global_inlier_ratio", 0.60)
        ),
        "max_global_inlier_rms_m": float(
            options.get("accept_max_global_inlier_rms_m", 0.006)
        ),
        "min_frame_inlier_ratio": float(
            options.get("accept_min_frame_inlier_ratio", 0.35)
        ),
        "max_frame_p95_residual_m": float(
            options.get("accept_max_frame_p95_residual_m", 0.015)
        ),
    }
    diagnostics = _plane_fit_diagnostics(
        fit_result,
        frame_samples,
        tolerance_m=tolerance_m,
        stride=stride,
        roi_uv=roi_uv,
        acceptance=acceptance,
    )
    if not diagnostics["quality_passed"]:
        failed_checks = [
            name
            for name, passed in diagnostics["quality_checks"].items()
            if not passed
        ]
        raise SessionValidationError(
            "table-plane calibration quality gate failed: "
            + ", ".join(failed_checks)
        )
    robot_transform = reader.metadata.robot_state.get("T_base_from_head")
    extrinsic = factory_extrinsics_to_transform(reader.metadata.depth_to_color)
    return nominal_calibration_from_config(
        nominal_config,
        E_color_from_depth=extrinsic,
        T_base_from_head=robot_transform,
        table_plane=fit_result.plane,
        state=CalibrationState.PLANE_CALIBRATED_PARTIAL,
        diagnostics={"table_plane_fit": diagnostics},
    )


__all__ = [
    "calibrate_table_plane_from_session",
    "factory_extrinsics_to_transform",
    "fit_empty_table_plane",
    "fit_empty_table_plane_result",
    "load_calibration",
    "load_json",
    "nominal_calibration_from_config",
    "save_calibration",
]
