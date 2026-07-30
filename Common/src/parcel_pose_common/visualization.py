"""Debug overlays for perception evidence; no control outputs."""

from __future__ import annotations


import numpy as np
from numpy.typing import ArrayLike, NDArray

from .models import CameraIntrinsics


ImageArray = NDArray[np.uint8]


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


__all__ = ["project_points_to_pixels"]
