"""Debug overlays for perception evidence; no control outputs."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .estimator import EstimationEvidence
from .models import CameraIntrinsics, PoseEstimate
from .projection import unproject_plane_points


ImageArray = NDArray[np.uint8]


def _cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenCV is required only for debug visualization") from exc
    return cv2


def _as_bgr(image: ArrayLike) -> ImageArray:
    array = np.asarray(image)
    if array.ndim == 2:
        return np.repeat(array[..., None], 3, axis=2).astype(np.uint8, copy=True)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"image must be HxW or HxWx3, got {array.shape}")
    return array.astype(np.uint8, copy=True)


def project_points_to_pixels(points_depth_m: ArrayLike, intrinsics: CameraIntrinsics) -> NDArray[np.float64]:
    points = np.asarray(points_depth_m, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, 3)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_depth_m must have shape (N, 3)")
    z = points[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = float(intrinsics.fx) * points[:, 0] / z + float(intrinsics.cx)
        v = float(intrinsics.fy) * points[:, 1] / z + float(intrinsics.cy)
    pixels = np.column_stack((u, v))
    pixels[~np.isfinite(pixels).all(axis=1) | (z <= 0.0)] = np.nan
    return pixels


def draw_pose_overlay(
    image_bgr: ArrayLike,
    estimate: PoseEstimate,
    *,
    evidence: EstimationEvidence | None = None,
    intrinsics: CameraIntrinsics | None = None,
) -> ImageArray:
    """Draw selected slab support, fitted footprint, and validity text."""

    cv2 = _cv2()
    output = _as_bgr(image_bgr)
    height, width = output.shape[:2]

    if evidence is not None and evidence.projection.count:
        pixels = evidence.projection.pixels_uv
        step = max(1, pixels.shape[0] // 4_000)
        sampled = np.rint(pixels[::step]).astype(np.int32)
        valid = (
            (sampled[:, 0] >= 0)
            & (sampled[:, 0] < width)
            & (sampled[:, 1] >= 0)
            & (sampled[:, 1] < height)
        )
        output[sampled[valid, 1], sampled[valid, 0]] = (80, 180, 80)

    if (
        evidence is not None
        and intrinsics is not None
        and evidence.rectangle.corners_xy_m is not None
    ):
        corners_3d = unproject_plane_points(
            evidence.rectangle.corners_xy_m,
            evidence.projection.plane,
            origin=evidence.projection.origin_3d_m,
            basis=(evidence.projection.basis_u_3d, evidence.projection.basis_v_3d),
        )
        pixels = project_points_to_pixels(corners_3d, intrinsics)
        if np.all(np.isfinite(pixels)):
            polygon = np.rint(pixels).astype(np.int32).reshape(-1, 1, 2)
            color = (40, 220, 255) if estimate.full_pose_valid else (0, 128, 255)
            cv2.polylines(output, [polygon], True, color, 2, cv2.LINE_AA)
            center = np.mean(pixels, axis=0)
            cv2.drawMarker(
                output,
                tuple(np.rint(center).astype(int)),
                color,
                cv2.MARKER_CROSS,
                14,
                2,
                cv2.LINE_AA,
            )

    yaw_text = "yaw: --"
    if estimate.yaw_mod_180_deg is not None:
        yaw_text = f"yaw: {estimate.yaw_mod_180_deg:.1f} deg"
    status = "valid" if estimate.full_pose_valid else "abstain"
    lines = [
        f"parcel pose: {status}",
        yaw_text,
        f"frame: {estimate.frame}",
        f"calibration: {estimate.calibration_state.value}",
    ]
    if estimate.reasons:
        lines.append("reason: " + ", ".join(estimate.reasons[:2]))
    y = 24
    for line in lines:
        cv2.putText(output, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(output, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (30, 30, 30), 1, cv2.LINE_AA)
        y += 21
    return output


def colorize_depth(depth_m: ArrayLike, *, min_m: float = 0.2, max_m: float = 1.5) -> ImageArray:
    """Create a deterministic visualization without altering metric depth."""

    cv2 = _cv2()
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("depth_m must be 2D")
    normalized = np.clip((depth - float(min_m)) / max(float(max_m) - float(min_m), 1e-9), 0.0, 1.0)
    gray = np.rint((1.0 - normalized) * 255.0).astype(np.uint8)
    gray[~np.isfinite(depth) | (depth <= 0.0)] = 0
    return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)


# Backwards-friendly short alias for callers producing replay overlays.
draw_estimate = draw_pose_overlay


__all__ = ["colorize_depth", "draw_estimate", "draw_pose_overlay", "project_points_to_pixels"]
