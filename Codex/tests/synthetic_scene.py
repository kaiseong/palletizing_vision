"""Deterministic metric scenes shared by the geometry acceptance tests.

The renderer deliberately works in the raw depth optical frame.  A pixel ray is
intersected with either the calibrated table plane or a configurable parcel-top
normal offset; there is no image-space scale approximation in the truth.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray


BOX_LONG_M = 0.400
BOX_SHORT_M = 0.253
BOX_HEIGHT_M = 0.160
DEPTH_SCALE_M = 0.001


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SyntheticIntrinsics:
    width: int = 640
    height: int = 480
    fx: float = 610.0
    fy: float = 608.0
    cx: float = 319.5
    cy: float = 239.5


@dataclass(frozen=True)
class SyntheticScene:
    intrinsics: SyntheticIntrinsics
    depth_m: NDArray[np.float32]
    depth_z16: NDArray[np.uint16]
    top_mask: NDArray[np.bool_]
    table_normal: FloatArray
    table_d: float
    top_normal: FloatArray
    top_d: float
    plane_u: FloatArray
    plane_v: FloatArray
    center_depth_m: FloatArray
    center_plane_xy_m: FloatArray
    yaw_rad: float
    box_long_m: float
    box_short_m: float
    box_height_m: float
    visible_top_points_depth_m: FloatArray
    visible_top_points_plane_xy_m: FloatArray
    visible_top_pixels_uv: FloatArray


def unit(vector: FloatArray) -> FloatArray:
    vector = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("cannot normalize a zero/non-finite vector")
    return vector / length


def plane_basis(normal: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return a deterministic right-handed basis spanning ``normal``'s plane."""

    n = unit(normal)
    seed = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(seed @ n)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = unit(seed - float(seed @ n) * n)
    v = unit(np.cross(n, u))
    return u, v


def line_angle_error_deg(actual_rad: float, expected_rad: float) -> float:
    """Smallest angular distance between unoriented lines, in degrees."""

    delta = (float(actual_rad) - float(expected_rad) + math.pi / 2.0) % math.pi - math.pi / 2.0
    return abs(math.degrees(delta))


def rectangle_support_points(
    *,
    center_xy_m: tuple[float, float] = (0.0, 0.0),
    yaw_deg: float = 23.0,
    points_per_edge: int = 160,
    interior_points: int = 1_200,
    noise_std_m: float = 0.0008,
    hole_rate: float = 0.0,
    outlier_count: int = 0,
    visible_sides: tuple[str, ...] = ("long_low", "long_high", "short_low", "short_high"),
    crop_bounds_xy_m: tuple[float, float, float, float] | None = None,
    long_m: float = BOX_LONG_M,
    short_m: float = BOX_SHORT_M,
    seed: int = 7,
) -> FloatArray:
    """Sample a metric rectangle in continuous plane coordinates.

    Side names describe the fixed-size box coordinates: ``long_low/high`` are
    the two short physical edges at +/- long/2; ``short_low/high`` are the two
    long physical edges at +/- short/2.
    """

    rng = np.random.default_rng(seed)
    long_m = float(long_m)
    short_m = float(short_m)
    if not long_m > short_m > 0.0:
        raise ValueError("rectangle dimensions must satisfy long_m > short_m > 0")
    along_long = np.array([math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))])
    along_short = np.array([-along_long[1], along_long[0]])
    center = np.asarray(center_xy_m, dtype=np.float64)
    edge_parts: list[FloatArray] = []
    long_t = np.linspace(-long_m / 2.0, long_m / 2.0, points_per_edge)
    short_t = np.linspace(-short_m / 2.0, short_m / 2.0, points_per_edge)

    if "long_low" in visible_sides:
        edge_parts.append(center - long_m / 2.0 * along_long + short_t[:, None] * along_short)
    if "long_high" in visible_sides:
        edge_parts.append(center + long_m / 2.0 * along_long + short_t[:, None] * along_short)
    if "short_low" in visible_sides:
        edge_parts.append(center - short_m / 2.0 * along_short + long_t[:, None] * along_long)
    if "short_high" in visible_sides:
        edge_parts.append(center + short_m / 2.0 * along_short + long_t[:, None] * along_long)

    if interior_points:
        a = rng.uniform(-long_m / 2.0, long_m / 2.0, interior_points)
        b = rng.uniform(-short_m / 2.0, short_m / 2.0, interior_points)
        edge_parts.append(center + a[:, None] * along_long + b[:, None] * along_short)

    points = np.concatenate(edge_parts, axis=0) if edge_parts else np.empty((0, 2), dtype=np.float64)
    if crop_bounds_xy_m is not None:
        xmin, xmax, ymin, ymax = crop_bounds_xy_m
        keep = (
            (points[:, 0] >= xmin)
            & (points[:, 0] <= xmax)
            & (points[:, 1] >= ymin)
            & (points[:, 1] <= ymax)
        )
        points = points[keep]
    if hole_rate:
        if not 0.0 <= hole_rate < 1.0:
            raise ValueError("hole_rate must be in [0, 1)")
        points = points[rng.random(points.shape[0]) >= hole_rate]
    if noise_std_m:
        points = points + rng.normal(0.0, noise_std_m, points.shape)
    if outlier_count:
        radial = rng.uniform([-0.35, -0.28], [0.35, 0.28], size=(outlier_count, 2))
        points = np.concatenate((points, center + radial), axis=0)
    return points.astype(np.float64, copy=False)


def orthographic_image_observation(
    points_xy_m: FloatArray,
    *,
    pixels_per_meter: float = 1_000.0,
    principal_uv: tuple[float, float] = (320.0, 240.0),
    image_shape: tuple[int, int] = (480, 640),
    border_margin_px: float = 1.5,
) -> tuple[FloatArray, FloatArray, tuple[int, int]]:
    """Map plane points to pixels and censor points outside an image rectangle.

    The map is used only to state which evidence is image-border censored.  All
    recovered dimensions and error assertions remain in metric plane space.
    Returned boundary coordinates are clipped to the actual image border so a
    fitter cannot accidentally reinterpret a crop boundary as a physical edge.
    """

    points = np.asarray(points_xy_m, dtype=np.float64)
    center = np.asarray(principal_uv, dtype=np.float64)
    pixels = center + pixels_per_meter * points
    height, width = image_shape
    keep = (
        (pixels[:, 0] >= -border_margin_px)
        & (pixels[:, 0] <= (width - 1) + border_margin_px)
        & (pixels[:, 1] >= -border_margin_px)
        & (pixels[:, 1] <= (height - 1) + border_margin_px)
    )
    observed_points = points[keep]
    observed_pixels = pixels[keep]
    observed_pixels[:, 0] = np.clip(observed_pixels[:, 0], 0.0, width - 1.0)
    observed_pixels[:, 1] = np.clip(observed_pixels[:, 1], 0.0, height - 1.0)
    return observed_points, observed_pixels, image_shape


def tilted_scene(
    *,
    yaw_deg: float = 23.0,
    center_plane_xy_m: tuple[float, float] = (0.0, 0.0),
    normal: tuple[float, float, float] = (0.08, -0.16, -0.984),
    table_point_depth_m: tuple[float, float, float] = (0.0, 0.0, 0.90),
    intrinsics: SyntheticIntrinsics = SyntheticIntrinsics(),
    box_long_m: float = BOX_LONG_M,
    box_short_m: float = BOX_SHORT_M,
    box_height_m: float = BOX_HEIGHT_M,
    depth_noise_std_m: float = 0.0008,
    hole_rate: float = 0.0,
    outlier_rate: float = 0.0,
    image_clip: tuple[int, int, int, int] | None = None,
    seed: int = 11,
) -> SyntheticScene:
    """Render a closed parcel top above a tilted table into a raw Z depth map."""

    rng = np.random.default_rng(seed)
    box_long_m = float(box_long_m)
    box_short_m = float(box_short_m)
    box_height_m = float(box_height_m)
    if not box_long_m > box_short_m > 0.0 or box_height_m <= 0.0:
        raise ValueError(
            "box dimensions must satisfy long_m > short_m > 0 and height_m > 0"
        )
    n = unit(np.asarray(normal, dtype=np.float64))
    table_point = np.asarray(table_point_depth_m, dtype=np.float64)
    # The normal must face the camera at the optical origin.
    if float(n @ -table_point) <= 0.0:
        n = -n
    table_d = float(n @ table_point)
    top_d = table_d + box_height_m
    u_axis, v_axis = plane_basis(n)
    center_plane = np.asarray(center_plane_xy_m, dtype=np.float64)
    top_reference = table_point + box_height_m * n
    center_depth = top_reference + center_plane[0] * u_axis + center_plane[1] * v_axis

    v_px, u_px = np.indices((intrinsics.height, intrinsics.width), dtype=np.float64)
    rays = np.stack(
        (
            (u_px - intrinsics.cx) / intrinsics.fx,
            (v_px - intrinsics.cy) / intrinsics.fy,
            np.ones_like(u_px),
        ),
        axis=-1,
    )
    top_denom = rays @ n
    table_denom = top_denom
    with np.errstate(divide="ignore", invalid="ignore"):
        top_t = top_d / top_denom
        table_t = table_d / table_denom
    top_points = rays * top_t[..., None]
    relative = top_points - center_depth
    theta = math.radians(yaw_deg)
    long_axis = math.cos(theta) * u_axis + math.sin(theta) * v_axis
    short_axis = -math.sin(theta) * u_axis + math.cos(theta) * v_axis
    long_coordinate = relative @ long_axis
    short_coordinate = relative @ short_axis
    top_mask = (
        (top_t > 0.0)
        & (np.abs(long_coordinate) <= box_long_m / 2.0)
        & (np.abs(short_coordinate) <= box_short_m / 2.0)
    )
    if image_clip is not None:
        x0, y0, x1, y1 = image_clip
        clip_mask = np.zeros_like(top_mask)
        clip_mask[max(0, y0) : min(intrinsics.height, y1), max(0, x0) : min(intrinsics.width, x1)] = True
        top_mask &= clip_mask

    depth_m = table_t.astype(np.float32)
    depth_m[top_mask] = top_t[top_mask].astype(np.float32)
    if depth_noise_std_m:
        valid = np.isfinite(depth_m) & (depth_m > 0.0)
        depth_m[valid] += rng.normal(0.0, depth_noise_std_m, int(valid.sum())).astype(np.float32)
    if hole_rate:
        holes = top_mask & (rng.random(top_mask.shape) < hole_rate)
        depth_m[holes] = 0.0
        top_mask = top_mask & ~holes
    if outlier_rate:
        outliers = top_mask & (rng.random(top_mask.shape) < outlier_rate)
        depth_m[outliers] += rng.uniform(-0.08, 0.08, int(outliers.sum())).astype(np.float32)

    valid_depth = np.isfinite(depth_m) & (depth_m > 0.0)
    depth_z16 = np.zeros(depth_m.shape, dtype=np.uint16)
    depth_z16[valid_depth] = np.rint(depth_m[valid_depth] / DEPTH_SCALE_M).clip(0, 65535).astype(np.uint16)
    visible_points = top_points[top_mask]
    visible_v, visible_u = np.nonzero(top_mask)
    visible_pixels_uv = np.stack((visible_u, visible_v), axis=-1).astype(np.float64)
    visible_relative = visible_points - center_depth
    visible_plane = np.stack((visible_relative @ u_axis, visible_relative @ v_axis), axis=-1) + center_plane

    return SyntheticScene(
        intrinsics=intrinsics,
        depth_m=depth_m,
        depth_z16=depth_z16,
        top_mask=top_mask,
        table_normal=n,
        table_d=table_d,
        top_normal=n.copy(),
        top_d=top_d,
        plane_u=u_axis,
        plane_v=v_axis,
        center_depth_m=center_depth,
        center_plane_xy_m=center_plane,
        yaw_rad=theta,
        box_long_m=box_long_m,
        box_short_m=box_short_m,
        box_height_m=box_height_m,
        visible_top_points_depth_m=visible_points,
        visible_top_points_plane_xy_m=visible_plane,
        visible_top_pixels_uv=visible_pixels_uv,
    )


def noisy_plane_points(
    *,
    normal: tuple[float, float, float] = (0.08, -0.16, -0.984),
    point: tuple[float, float, float] = (0.0, 0.0, 0.90),
    inlier_count: int = 800,
    outlier_count: int = 100,
    noise_std_m: float = 0.0007,
    seed: int = 19,
) -> tuple[FloatArray, FloatArray, float, FloatArray]:
    """Return points, oriented truth normal, truth d, and the inlier mask."""

    rng = np.random.default_rng(seed)
    n = unit(np.asarray(normal, dtype=np.float64))
    p0 = np.asarray(point, dtype=np.float64)
    if float(n @ -p0) <= 0.0:
        n = -n
    u_axis, v_axis = plane_basis(n)
    a = rng.uniform(-0.45, 0.45, inlier_count)
    b = rng.uniform(-0.32, 0.32, inlier_count)
    inliers = p0 + a[:, None] * u_axis + b[:, None] * v_axis
    inliers += rng.normal(0.0, noise_std_m, (inlier_count, 1)) * n
    outliers = rng.uniform([-0.45, -0.32, 0.45], [0.45, 0.32, 1.20], (outlier_count, 3))
    points = np.concatenate((inliers, outliers), axis=0)
    mask = np.zeros(points.shape[0], dtype=np.bool_)
    mask[:inlier_count] = True
    permutation = rng.permutation(points.shape[0])
    return points[permutation], n, float(n @ p0), mask[permutation]
