"""Camera geometry: intrinsics, deprojection, and RANSAC support-plane fitting.

Everything here is colour-agnostic and depth-driven. The support plane is the
surface the box rests on (desk for picking, pallet / lower box layer for
placing). Its normal is always oriented *toward the camera* so that points that
stick up toward the lens (the box top) have a positive signed height above it.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


# --------------------------------------------------------------------------- #
# Intrinsics
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int | None = None
    height: int | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "CameraIntrinsics":
        if "camera_matrix" in data:
            m = np.asarray(data["camera_matrix"], dtype=np.float64)
            return cls(float(m[0, 0]), float(m[1, 1]), float(m[0, 2]), float(m[1, 2]))
        return cls(
            float(data["fx"]),
            float(data["fy"]),
            float(data.get("cx", data.get("ppx"))),
            float(data.get("cy", data.get("ppy"))),
            int(data["width"]) if data.get("width") is not None else None,
            int(data["height"]) if data.get("height") is not None else None,
        )

    def project(self, points_xyz: np.ndarray) -> np.ndarray:
        """Pinhole-project camera-frame 3D points to (u, v) pixels."""
        p = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
        u = self.fx * p[:, 0] / p[:, 2] + self.cx
        v = self.fy * p[:, 1] / p[:, 2] + self.cy
        return np.column_stack((u, v))


# --------------------------------------------------------------------------- #
# Small vector helpers
# --------------------------------------------------------------------------- #
def normalize(vector: np.ndarray, *, eps: float = 1e-9) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm < eps:
        raise ValueError("Cannot normalize a near-zero vector.")
    return arr / norm


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal in-plane axes (e1, e2) for a plane with the given normal."""
    n = normalize(normal)
    seed = np.array([1.0, 0.0, 0.0]) if abs(float(n[0])) <= 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = normalize(seed - float(seed @ n) * n)
    e2 = normalize(np.cross(n, e1))
    return e1, e2


def even_subsample(count: int, limit: int) -> np.ndarray:
    """Evenly spaced index subset, or all indices when count <= limit."""
    if count <= limit:
        return np.arange(count)
    return np.linspace(0, count - 1, limit).astype(np.int64)


# --------------------------------------------------------------------------- #
# Deprojection
# --------------------------------------------------------------------------- #
def deproject_valid(
    depth_m: np.ndarray,
    intr: CameraIntrinsics,
    *,
    z_min: float = 0.05,
    z_max: float = 3.0,
    roi_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Back-project every valid depth pixel in [z_min, z_max] to camera-frame XYZ.

    Returns (points Nx3 float64, rows N int, cols N int) so callers can carry
    pixel indices through segmentation without re-deprojecting.
    """
    depth = np.asarray(depth_m, dtype=np.float64)
    valid = np.isfinite(depth) & (depth >= float(z_min)) & (depth <= float(z_max))
    if roi_mask is not None:
        valid &= np.asarray(roi_mask) > 0
    rows, cols = np.nonzero(valid)
    z = depth[rows, cols]
    x = (cols.astype(np.float64) - intr.cx) * z / intr.fx
    y = (rows.astype(np.float64) - intr.cy) * z / intr.fy
    return np.column_stack((x, y, z)), rows, cols


# --------------------------------------------------------------------------- #
# Support plane
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SupportPlane:
    normal: np.ndarray  # unit, oriented toward the camera (normal . point < 0)
    point: np.ndarray   # a point on the plane (camera frame, metres)
    inlier_fraction: float = 0.0
    rms_m: float = 0.0

    def signed_height(self, points: np.ndarray) -> np.ndarray:
        """Signed distance from the plane; > 0 means toward the camera (box top)."""
        return (np.asarray(points, dtype=np.float64).reshape(-1, 3) - self.point) @ self.normal


def _orient_toward_camera(normal: np.ndarray, point: np.ndarray) -> np.ndarray:
    # Camera sits at the origin; toward-camera means the plane point is on the
    # negative side of the normal.
    return -normal if float(normal @ point) > 0.0 else normal


def fit_support_plane(
    points: np.ndarray,
    *,
    iterations: int = 250,
    tolerance_m: float = 0.006,
    min_inliers: int = 100,
    rng: np.random.Generator | None = None,
) -> SupportPlane | None:
    """RANSAC the dominant plane of a point set (the support surface)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = pts.shape[0]
    if n < 3:
        return None
    rng = rng or np.random.default_rng(12345)

    best_inliers: np.ndarray | None = None
    best_count = -1
    for _ in range(int(iterations)):
        idx = rng.choice(n, size=3, replace=False)
        p1, p2, p3 = pts[idx]
        normal = np.cross(p2 - p1, p3 - p1)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            continue
        normal /= norm
        dist = np.abs((pts - p1) @ normal)
        inliers = dist <= tolerance_m
        count = int(np.count_nonzero(inliers))
        if count > best_count:
            best_count = count
            best_inliers = inliers
    if best_inliers is None or best_count < min_inliers:
        return None

    inlier_pts = pts[best_inliers]
    origin = inlier_pts.mean(axis=0)
    _, _, vh = np.linalg.svd(inlier_pts - origin, full_matrices=False)
    normal = _orient_toward_camera(normalize(vh[-1]), origin)
    rms = float(np.sqrt(np.mean(((inlier_pts - origin) @ normal) ** 2)))
    return SupportPlane(normal, origin, float(best_count / n), rms)


def refine_plane(
    plane: SupportPlane,
    points: np.ndarray,
    *,
    band_m: float = 0.010,
    max_angle_deg: float = 4.0,
    max_offset_m: float = 0.015,
) -> SupportPlane:
    """Re-fit the plane to nearby points to absorb small base tilt / vibration.

    The refinement is rejected (original plane kept) when it drifts too far,
    which would mean it latched onto a different surface.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    signed = plane.signed_height(pts)
    gated = np.abs(signed) <= band_m
    if int(np.count_nonzero(gated)) < 100:
        return plane

    sub = pts[gated]
    origin = sub.mean(axis=0)
    _, _, vh = np.linalg.svd(sub - origin, full_matrices=False)
    normal = normalize(vh[-1])
    if float(normal @ plane.normal) < 0.0:
        normal = -normal
    angle = math.degrees(math.acos(min(max(float(normal @ plane.normal), -1.0), 1.0)))
    offset = abs(float((origin - plane.point) @ plane.normal))
    if angle > max_angle_deg or offset > max_offset_m:
        return plane

    normal = _orient_toward_camera(normal, origin)
    rms = float(np.sqrt(np.mean(((sub - origin) @ normal) ** 2)))
    return SupportPlane(normal, origin, plane.inlier_fraction, rms)
