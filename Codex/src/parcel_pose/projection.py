"""Raw-depth deprojection and exact calibrated-plane projection."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .models import CameraIntrinsics, Plane
from .plane import point_on_plane


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PlaneProjection:
    """Metric evidence selected by depth and intersected with an exact plane."""

    points_3d_m: FloatArray
    points_xy_m: FloatArray
    pixels_uv: FloatArray
    source_depth_m: FloatArray
    mask: NDArray[np.bool_]
    plane: Plane
    origin_3d_m: FloatArray
    basis_u_3d: FloatArray
    basis_v_3d: FloatArray
    diagnostics: dict[str, Any]

    @property
    def count(self) -> int:
        return int(self.points_xy_m.shape[0])


def _validate_intrinsics(intrinsics: CameraIntrinsics) -> None:
    values = (intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("camera intrinsics must be finite")
    if float(intrinsics.fx) <= 0.0 or float(intrinsics.fy) <= 0.0:
        raise ValueError("camera focal lengths must be positive")
    if any(abs(float(value)) > 1e-12 for value in intrinsics.coeffs):
        raise ValueError(
            "raw-depth intrinsics contain non-zero distortion coefficients; "
            "this vectorized D435 path requires the rectified zero-distortion "
            "depth profile or SDK-equivalent deprojection"
        )


def depth_to_meters(depth: ArrayLike, depth_scale: float | None = None) -> FloatArray:
    """Convert raw Z16/integer depth or metric floating depth to meters."""

    array = np.asarray(depth)
    if array.ndim != 2:
        raise ValueError(f"depth must have shape (H, W), got {array.shape}")
    if np.issubdtype(array.dtype, np.integer):
        if depth_scale is None:
            raise ValueError("depth_scale is required for integer/raw depth")
        scale = float(depth_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("depth_scale must be positive and finite")
        return array.astype(np.float64) * scale
    if depth_scale is not None and not math.isclose(float(depth_scale), 1.0):
        # Explicit scaling of floats is allowed for recorded numeric arrays,
        # but metric callers should normally omit the scale.
        scale = float(depth_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("depth_scale must be positive and finite")
        return array.astype(np.float64) * scale
    return array.astype(np.float64, copy=False)


def intersect_rays_with_plane(
    rays: ArrayLike,
    plane: Plane,
    *,
    ray_origin: ArrayLike = (0.0, 0.0, 0.0),
    min_denominator: float = 1e-9,
) -> tuple[FloatArray, NDArray[np.bool_]]:
    """Intersect rays with ``plane`` and reject parallel/behind-origin hits."""

    directions = np.asarray(rays, dtype=np.float64)
    if directions.ndim == 1:
        directions = directions.reshape(1, 3)
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError(f"rays must have shape (N, 3), got {directions.shape}")
    origin = np.asarray(ray_origin, dtype=np.float64).reshape(3)
    normal = np.asarray(plane.normal, dtype=np.float64).reshape(3)
    denominators = directions @ normal
    numer = float(plane.d) - float(normal @ origin)
    with np.errstate(divide="ignore", invalid="ignore"):
        distances = numer / denominators
    valid = (
        np.all(np.isfinite(directions), axis=1)
        & np.isfinite(distances)
        & (np.abs(denominators) > float(min_denominator))
        & (distances > 0.0)
    )
    intersections = np.full_like(directions, np.nan, dtype=np.float64)
    intersections[valid] = origin + distances[valid, None] * directions[valid]
    return intersections, valid


def plane_basis(normal_or_plane: ArrayLike | Plane) -> tuple[FloatArray, FloatArray]:
    """Return a deterministic right-handed orthonormal basis for a plane."""

    raw = normal_or_plane.normal if isinstance(normal_or_plane, Plane) else normal_or_plane
    normal = np.array(raw, dtype=np.float64, copy=True).reshape(3)
    norm = float(np.linalg.norm(normal))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("normal must be finite and non-zero")
    normal /= norm
    seed = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(seed @ normal)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    basis_u = seed - float(seed @ normal) * normal
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(normal, basis_u)
    basis_v /= np.linalg.norm(basis_v)
    return basis_u, basis_v


def project_points_to_plane(
    points: ArrayLike,
    plane: Plane,
    *,
    origin: ArrayLike | None = None,
    basis: tuple[ArrayLike, ArrayLike] | None = None,
) -> FloatArray:
    """Express 3D points as continuous metric coordinates on ``plane``."""

    cloud = np.asarray(points, dtype=np.float64)
    if cloud.ndim == 1:
        cloud = cloud.reshape(1, 3)
    if cloud.ndim != 2 or cloud.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {cloud.shape}")
    anchor = point_on_plane(plane) if origin is None else np.asarray(origin, dtype=np.float64).reshape(3)
    basis_u, basis_v = plane_basis(plane) if basis is None else basis
    u = np.asarray(basis_u, dtype=np.float64).reshape(3)
    v = np.asarray(basis_v, dtype=np.float64).reshape(3)
    relative = cloud - anchor
    return np.column_stack((relative @ u, relative @ v))


def unproject_plane_points(
    points_xy: ArrayLike,
    plane: Plane,
    *,
    origin: ArrayLike | None = None,
    basis: tuple[ArrayLike, ArrayLike] | None = None,
) -> FloatArray:
    """Lift continuous metric plane coordinates back to 3D."""

    coordinates = np.asarray(points_xy, dtype=np.float64)
    if coordinates.ndim == 1:
        coordinates = coordinates.reshape(1, 2)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(f"points_xy must have shape (N, 2), got {coordinates.shape}")
    anchor = point_on_plane(plane) if origin is None else np.asarray(origin, dtype=np.float64).reshape(3)
    basis_u, basis_v = plane_basis(plane) if basis is None else basis
    return (
        anchor
        + coordinates[:, 0, None] * np.asarray(basis_u, dtype=np.float64)
        + coordinates[:, 1, None] * np.asarray(basis_v, dtype=np.float64)
    )


class DepthPlaneProjector:
    """Project many depth frames through one fixed camera/plane geometry.

    Intrinsics and the calibrated top plane do not change while the RB-Y1
    torso/head and camera mount remain fixed.  Cache their pixel rays and
    plane terms once, then build 3D points only for the selected slab pixels.
    """

    def __init__(self, intrinsics: CameraIntrinsics, plane: Plane) -> None:
        _validate_intrinsics(intrinsics)
        self.intrinsics = intrinsics
        self.plane = plane

        cols = np.arange(int(intrinsics.width), dtype=np.float64)
        rows = np.arange(int(intrinsics.height), dtype=np.float64)
        self._ray_x = (cols - float(intrinsics.cx)) / float(intrinsics.fx)
        self._ray_y = (rows - float(intrinsics.cy)) / float(intrinsics.fy)
        normal = np.asarray(plane.normal, dtype=np.float64)
        self._plane_ray_denominator = (
            self._ray_x[None, :] * normal[0]
            + self._ray_y[:, None] * normal[1]
            + normal[2]
        )
        self._origin = point_on_plane(plane)
        self._basis_u, self._basis_v = plane_basis(plane)
        for cached in (
            self._ray_x,
            self._ray_y,
            self._plane_ray_denominator,
            self._origin,
            self._basis_u,
            self._basis_v,
        ):
            cached.setflags(write=False)

    def project(
        self,
        depth: ArrayLike,
        *,
        depth_scale: float | None = None,
        slab_tolerance_m: float = 0.020,
        min_depth_m: float = 0.20,
        max_depth_m: float = 2.0,
        support_mask: ArrayLike | None = None,
        max_points: int | None = 50_000,
    ) -> PlaneProjection:
        """Gate and intersect one frame using the cached fixed geometry."""

        tolerance = float(slab_tolerance_m)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("slab_tolerance_m must be positive and finite")
        depth_m = depth_to_meters(depth, depth_scale)
        expected_shape = (self.intrinsics.height, self.intrinsics.width)
        if depth_m.shape != expected_shape:
            raise ValueError(
                "depth shape does not match raw-depth intrinsics: "
                f"{depth_m.shape} vs {expected_shape}"
            )
        valid_depth = (
            np.isfinite(depth_m)
            & (depth_m >= float(min_depth_m))
            & (depth_m <= float(max_depth_m))
        )
        distances = depth_m * self._plane_ray_denominator - float(self.plane.d)
        mask = valid_depth & np.isfinite(distances) & (np.abs(distances) <= tolerance)
        if support_mask is not None:
            support = np.asarray(support_mask, dtype=np.bool_)
            if support.shape != mask.shape:
                raise ValueError(
                    f"support_mask shape {support.shape} does not match depth {mask.shape}"
                )
            mask &= support

        rows, cols = np.nonzero(mask)
        raw_count = int(rows.size)
        if max_points is not None and rows.size > int(max_points):
            indices = np.linspace(0, rows.size - 1, int(max_points), dtype=np.int64)
            rows = rows[indices]
            cols = cols[indices]
        pixels = np.column_stack((cols, rows)).astype(np.float64)
        rays = np.column_stack(
            (
                self._ray_x[cols],
                self._ray_y[rows],
                np.ones(rows.size, dtype=np.float64),
            )
        )
        intersections, ray_valid = intersect_rays_with_plane(rays, self.plane)
        pixels = pixels[ray_valid]
        intersections = intersections[ray_valid]
        selected_depth = depth_m[rows, cols][ray_valid]
        coordinates = project_points_to_plane(
            intersections,
            self.plane,
            origin=self._origin,
            basis=(self._basis_u, self._basis_v),
        )
        return PlaneProjection(
            points_3d_m=intersections,
            points_xy_m=coordinates,
            pixels_uv=pixels,
            source_depth_m=selected_depth,
            mask=mask,
            plane=self.plane,
            origin_3d_m=self._origin,
            basis_u_3d=self._basis_u,
            basis_v_3d=self._basis_v,
            diagnostics={
                "valid_depth_pixels": int(np.count_nonzero(valid_depth)),
                "slab_pixels": raw_count,
                "projected_points": int(coordinates.shape[0]),
                "slab_tolerance_m": tolerance,
                "depth_range_m": [float(min_depth_m), float(max_depth_m)],
            },
        )

