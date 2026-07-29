"""Plane fitting primitives used by the metric parcel estimator.

The package has one plane convention everywhere: ``normal @ point == d``.
Normals are unit length.  A calibrated table normal is oriented toward the
camera/parcel side so a positive offset moves from the table to the parcel
top, including when the camera or table is tilted.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .models import Plane


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PlaneFitResult:
    """A robust plane and the evidence used to accept it."""

    plane: Plane
    inlier_mask: NDArray[np.bool_]
    residuals_m: FloatArray
    inlier_rms_m: float
    inlier_ratio: float
    iterations: int

    @property
    def inlier_count(self) -> int:
        return int(np.count_nonzero(self.inlier_mask))

    def to_dict(self) -> dict[str, object]:
        return {
            "plane": {
                "normal": np.asarray(self.plane.normal, dtype=np.float64).tolist(),
                "d": float(self.plane.d),
                "frame": self.plane.frame,
            },
            "inlier_count": self.inlier_count,
            "sample_count": int(self.inlier_mask.size),
            "inlier_ratio": self.inlier_ratio,
            "inlier_rms_m": self.inlier_rms_m,
            "iterations": self.iterations,
        }


def _finite_points(points: ArrayLike, *, minimum: int = 3) -> FloatArray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {array.shape}")
    array = array[np.all(np.isfinite(array), axis=1)]
    if array.shape[0] < minimum:
        raise ValueError(f"at least {minimum} finite 3D points are required")
    return array


def _unit(vector: ArrayLike) -> FloatArray:
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("plane normal must be finite and non-zero")
    return value / norm


def make_plane(normal: ArrayLike, d: float, *, frame: str = "depth") -> Plane:
    """Create a normalized plane without changing its geometric locus."""

    raw = np.asarray(normal, dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(raw))
    if not math.isfinite(length) or length <= 1e-12:
        raise ValueError("plane normal must be finite and non-zero")
    scalar = float(d)
    if not math.isfinite(scalar):
        raise ValueError("plane d must be finite")
    return Plane(normal=raw / length, d=scalar / length, frame=frame)


def point_on_plane(plane: Plane) -> FloatArray:
    """Return the point on ``plane`` closest to the frame origin."""

    normal = _unit(plane.normal)
    return normal * float(plane.d)


def signed_distances(points: ArrayLike, plane: Plane) -> FloatArray:
    """Return signed orthogonal distances because ``plane.normal`` is unit."""

    array = np.asarray(points, dtype=np.float64)
    if array.shape[-1] != 3:
        raise ValueError(f"points must end in dimension 3, got {array.shape}")
    return array @ _unit(plane.normal) - float(plane.d)


def orient_plane_toward(
    plane: Plane,
    camera_origin: ArrayLike = (0.0, 0.0, 0.0),
    *,
    point: ArrayLike | None = None,
) -> Plane:
    """Orient a plane normal toward the camera/parcel side.

    The required invariant is ``n @ (camera_origin - point_on_plane) > 0``.
    ``point`` may be supplied when a particular calibrated point should be
    used for the sign check; otherwise the closest point is equivalent.
    """

    normal = _unit(plane.normal)
    scalar = float(plane.d)
    camera = np.asarray(camera_origin, dtype=np.float64).reshape(3)
    plane_point = point_on_plane(plane) if point is None else np.asarray(point, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(camera)) or not np.all(np.isfinite(plane_point)):
        raise ValueError("camera_origin and point must be finite")
    if float(normal @ (camera - plane_point)) <= 0.0:
        normal = -normal
        scalar = -scalar
    return Plane(normal=normal, d=scalar, frame=plane.frame)


def offset_plane(plane: Plane, offset_m: float, *, frame: str | None = None) -> Plane:
    """Offset ``plane`` along its oriented unit normal by ``offset_m``."""

    offset = float(offset_m)
    if not math.isfinite(offset):
        raise ValueError("offset_m must be finite")
    normalized = make_plane(plane.normal, plane.d, frame=plane.frame)
    return Plane(
        normal=np.asarray(normalized.normal, dtype=np.float64).copy(),
        d=float(normalized.d) + offset,
        frame=normalized.frame if frame is None else frame,
    )


def fit_plane_svd(
    points: ArrayLike,
    *,
    frame: str = "depth",
    camera_origin: ArrayLike | None = None,
) -> Plane:
    """Fit a least-squares plane to finite points with deterministic SVD."""

    cloud = _finite_points(points)
    centroid = np.mean(cloud, axis=0)
    _, singular_values, vh = np.linalg.svd(cloud - centroid, full_matrices=False)
    if singular_values.size < 2 or float(singular_values[1]) <= 1e-12:
        raise ValueError("points are collinear; a plane is not identifiable")
    normal = _unit(vh[-1])
    fitted = Plane(normal=normal, d=float(normal @ centroid), frame=frame)
    if camera_origin is not None:
        fitted = orient_plane_toward(fitted, camera_origin, point=centroid)
    return fitted


def fit_plane_ransac(
    points: ArrayLike,
    *,
    tolerance_m: float = 0.004,
    iterations: int = 300,
    min_inliers: int = 50,
    min_inlier_ratio: float = 0.35,
    seed: int = 1729,
    frame: str = "depth",
    camera_origin: ArrayLike | None = (0.0, 0.0, 0.0),
) -> PlaneFitResult:
    """Fit a deterministic RANSAC plane, then refine it over all inliers."""

    cloud = _finite_points(points)
    tolerance = float(tolerance_m)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance_m must be positive and finite")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    required = max(3, int(min_inliers), int(math.ceil(float(min_inlier_ratio) * cloud.shape[0])))
    rng = np.random.default_rng(seed)

    best_mask: NDArray[np.bool_] | None = None
    best_count = -1
    best_median = math.inf
    for _ in range(int(iterations)):
        indices = rng.choice(cloud.shape[0], size=3, replace=False)
        a, b, c = cloud[indices]
        raw_normal = np.cross(b - a, c - a)
        length = float(np.linalg.norm(raw_normal))
        if length <= 1e-10:
            continue
        normal = raw_normal / length
        residuals = np.abs(cloud @ normal - float(normal @ a))
        mask = residuals <= tolerance
        count = int(np.count_nonzero(mask))
        if count < 3:
            continue
        median = float(np.median(residuals[mask]))
        if count > best_count or (count == best_count and median < best_median):
            best_count = count
            best_median = median
            best_mask = mask

    if best_mask is None or best_count < required:
        raise ValueError(
            f"plane RANSAC found {max(best_count, 0)} inliers; at least {required} are required"
        )

    fitted = fit_plane_svd(
        cloud[best_mask],
        frame=frame,
        camera_origin=camera_origin,
    )
    residuals = np.abs(signed_distances(cloud, fitted))
    final_mask = residuals <= tolerance
    if int(np.count_nonzero(final_mask)) >= 3:
        fitted = fit_plane_svd(
            cloud[final_mask],
            frame=frame,
            camera_origin=camera_origin,
        )
        residuals = np.abs(signed_distances(cloud, fitted))
        final_mask = residuals <= tolerance

    count = int(np.count_nonzero(final_mask))
    rms = float(np.sqrt(np.mean(np.square(residuals[final_mask])))) if count else math.inf
    return PlaneFitResult(
        plane=fitted,
        inlier_mask=final_mask,
        residuals_m=residuals,
        inlier_rms_m=rms,
        inlier_ratio=float(count / cloud.shape[0]),
        iterations=int(iterations),
    )


