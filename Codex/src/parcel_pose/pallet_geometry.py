"""Metric RGB-D estimator for the RB-Y1 pallet slot-1 hover primitive.

The estimator uses depth to discover the stack top plane and the metric void
inside the four-box ring.  RGB is associated evidence only; it never supplies
metric scale.  A closer held-box plane is explicitly excluded before the stack
opening is recovered.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .models import CameraIntrinsics, Plane
from .pallet_models import (
    BoundaryLineEvidence,
    HeldBoxHint,
    HeldBoxTopObservation,
    LCornerObservation,
    PalletEstimatorConfig,
    PalletFrameEvidence,
    PalletSceneObservation,
    StackObservation,
)
from .transforms import validate_transform


FloatArray = NDArray[np.float64]
ImageArray = NDArray[np.uint8]


def _cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for pallet rim fitting") from exc
    return cv2


def _unit(vector: ArrayLike) -> FloatArray:
    result = np.asarray(vector, dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(result))
    if not math.isfinite(length) or length <= 1e-9:
        raise ValueError("direction must be finite and non-zero")
    return result / length


def _line_angle_difference(a: ArrayLike, b: ArrayLike) -> float:
    first = np.asarray(a, dtype=np.float64)
    second = np.asarray(b, dtype=np.float64)
    cosine = float(np.clip(abs(first @ second), 0.0, 1.0))
    return math.acos(cosine)


@dataclass(frozen=True, slots=True)
class _PlaneCandidate:
    plane: Plane
    seed_count: int
    inlier_count: int
    p95_residual_m: float
    uncertainty_m: float


@dataclass(frozen=True, slots=True)
class _OpeningCandidate:
    label: int
    box_grid: FloatArray
    center_grid: FloatArray
    size_sorted_m: tuple[float, float]
    dimension_error_m: tuple[float, float]
    score: float


@dataclass(frozen=True, slots=True)
class _LineFit2D:
    centroid_xy: FloatArray
    direction_xy: FloatArray
    endpoints_xy: FloatArray
    support_point_count: int
    support_length_m: float
    p95_residual_m: float
    axis_residual_rad: float


@dataclass(frozen=True, slots=True)
class _LCornerFitResult:
    observation: LCornerObservation
    component_boundary_base: FloatArray | None


def _fit_trimmed_horizontal_plane(
    points: FloatArray,
    *,
    slab_m: float,
    max_points: int,
) -> _PlaneCandidate:
    cloud = np.asarray(points, dtype=np.float64)
    cloud = cloud[np.all(np.isfinite(cloud), axis=1)]
    if cloud.shape[0] < 100:
        raise ValueError("fewer than 100 finite points support the plane")
    if cloud.shape[0] > int(max_points):
        indices = np.linspace(0, cloud.shape[0] - 1, int(max_points), dtype=np.int64)
        cloud = cloud[indices]
    retained = cloud
    normal = np.array((0.0, 0.0, 1.0), dtype=np.float64)
    scalar = float(np.median(cloud[:, 2]))
    for tolerance in (1.5 * slab_m, slab_m, 0.67 * slab_m):
        if retained.shape[0] < 100:
            break
        centroid = np.mean(retained, axis=0)
        _, singular_values, vh = np.linalg.svd(retained - centroid, full_matrices=False)
        if singular_values.size < 2 or float(singular_values[1]) <= 1e-10:
            raise ValueError("plane support is collinear")
        normal = vh[-1]
        if normal[2] < 0.0:
            normal = -normal
        scalar = float(normal @ centroid)
        residuals = np.abs(cloud @ normal - scalar)
        retained = cloud[residuals <= tolerance]
    if retained.shape[0] < 100:
        raise ValueError("plane trimming removed too much support")
    centroid = np.mean(retained, axis=0)
    _, _, vh = np.linalg.svd(retained - centroid, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0.0:
        normal = -normal
    scalar = float(normal @ centroid)
    residuals = np.abs(cloud @ normal - scalar)
    inliers = residuals <= slab_m
    values = residuals[inliers]
    if values.size < 100:
        raise ValueError("plane has insufficient final inliers")
    return _PlaneCandidate(
        plane=Plane(normal=normal, d=scalar, frame="base"),
        seed_count=int(cloud.shape[0]),
        inlier_count=int(values.size),
        p95_residual_m=float(np.percentile(values, 95)),
        uncertainty_m=float(np.median(np.abs(values - np.median(values))) * 1.4826),
    )


def _dominant_height(
    heights: FloatArray,
    *,
    lower_m: float,
    upper_m: float,
    bin_m: float,
) -> tuple[float, int]:
    selected = heights[
        np.isfinite(heights) & (heights >= float(lower_m)) & (heights <= float(upper_m))
    ]
    if selected.size < 100:
        raise ValueError("insufficient heights in the requested range")
    edges = np.arange(float(lower_m), float(upper_m) + float(bin_m), float(bin_m))
    counts, edges = np.histogram(selected, bins=edges)
    if counts.size == 0 or int(np.max(counts)) < 30:
        raise ValueError("no dominant horizontal support")
    # This estimator's workspace contract contains one pallet stack after the
    # held-carton footprint has been excluded.  Select the dominant support,
    # not merely the highest supported bin: recorded partial-stack scenes can
    # contain sparse upper clutter whose plane is not the stack top.
    index = int(np.argmax(counts))
    return float(0.5 * (edges[index] + edges[index + 1])), int(counts[index])


def _kernel_size(distance_m: float, resolution_m: float) -> int:
    value = max(1, int(round(float(distance_m) / float(resolution_m))))
    return value if value % 2 == 1 else value + 1


def _grid_to_base_xy(
    points_grid: FloatArray,
    *,
    x_max: float,
    y_max: float,
    resolution_m: float,
) -> FloatArray:
    # OpenCV points are (column, row).  Increasing image column follows
    # camera image-right/base -Y; increasing row follows base -X.
    return np.column_stack(
        (
            float(x_max) - points_grid[:, 1] * float(resolution_m),
            float(y_max) - points_grid[:, 0] * float(resolution_m),
        )
    )


def _plane_z(plane: Plane, xy: ArrayLike) -> FloatArray:
    points = np.asarray(xy, dtype=np.float64)
    normal = np.asarray(plane.normal, dtype=np.float64)
    if abs(float(normal[2])) <= 1e-6:
        raise ValueError("stack plane is vertical")
    return (
        float(plane.d) - normal[0] * points[..., 0] - normal[1] * points[..., 1]
    ) / normal[2]


def _opening_candidates(
    labels: NDArray[np.int32],
    stats: NDArray[np.int32],
    *,
    resolution_m: float,
    expected_m: tuple[float, float],
    minimum_m: float,
    maximum_m: float,
) -> list[_OpeningCandidate]:
    cv2 = _cv2()
    result: list[_OpeningCandidate] = []
    expected_sorted = np.sort(np.asarray(expected_m, dtype=np.float64))
    max_area_m2 = 1.5 * float(maximum_m) ** 2
    min_area_m2 = 0.55 * float(minimum_m) ** 2
    for label in range(1, int(stats.shape[0])):
        area_m2 = float(stats[label, cv2.CC_STAT_AREA]) * resolution_m**2
        if not (min_area_m2 <= area_m2 <= max_area_m2):
            continue
        left = int(stats[label, cv2.CC_STAT_LEFT])
        top = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        component_labels = labels[top : top + height, left : left + width]
        rows, cols = np.nonzero(component_labels == label)
        if rows.size < 20:
            continue
        rectangle = cv2.minAreaRect(
            np.column_stack((cols + left, rows + top)).astype(np.float32)
        )
        sizes = np.sort(np.asarray(rectangle[1], dtype=np.float64) * resolution_m)
        if sizes[0] < minimum_m or sizes[1] > maximum_m:
            continue
        errors = np.abs(sizes - expected_sorted)
        fill_ratio = area_m2 / max(float(sizes[0] * sizes[1]), 1e-9)
        score = float(np.sum(errors) + 0.020 * abs(1.0 - min(fill_ratio, 1.0)))
        result.append(
            _OpeningCandidate(
                label=label,
                box_grid=np.asarray(cv2.boxPoints(rectangle), dtype=np.float64),
                center_grid=np.asarray(rectangle[0], dtype=np.float64),
                size_sorted_m=(float(sizes[0]), float(sizes[1])),
                dimension_error_m=(float(errors[0]), float(errors[1])),
                score=score,
            )
        )
    result.sort(key=lambda candidate: (candidate.score, candidate.label))
    return result


def _side_evidence(
    occupancy: NDArray[np.uint8],
    component: NDArray[np.uint8],
    box_grid: FloatArray,
    *,
    resolution_m: float,
    outer_band_m: tuple[float, float],
    min_support_ratio: float,
) -> tuple[tuple[float, ...], tuple[bool, ...], float]:
    cv2 = _cv2()
    center = np.mean(box_grid, axis=0)
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    boundary = (
        max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
        if contours
        else np.empty((0, 2), dtype=np.float64)
    )
    support_ratios: list[float] = []
    observed: list[bool] = []
    fitted_directions: list[FloatArray | None] = []
    inner_px = max(1.0, float(outer_band_m[0]) / resolution_m)
    outer_px = max(inner_px + 1.0, float(outer_band_m[1]) / resolution_m)
    height, width = occupancy.shape
    for index in range(4):
        start = box_grid[index]
        stop = box_grid[(index + 1) % 4]
        edge = stop - start
        length = float(np.linalg.norm(edge))
        if length <= 2.0:
            support_ratios.append(0.0)
            observed.append(False)
            fitted_directions.append(None)
            continue
        direction = edge / length
        midpoint = 0.5 * (start + stop)
        outward = midpoint - center
        outward /= max(float(np.linalg.norm(outward)), 1e-9)
        along = np.linspace(0.08, 0.92, max(16, int(length)), dtype=np.float64)
        offsets = np.linspace(inner_px, outer_px, max(4, int(outer_px - inner_px) + 1))
        samples = (
            start[None, None, :]
            + along[:, None, None] * edge[None, None, :]
            + offsets[None, :, None] * outward[None, None, :]
        ).reshape(-1, 2)
        cols = np.rint(samples[:, 0]).astype(np.int64)
        rows = np.rint(samples[:, 1]).astype(np.int64)
        inside = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
        ratio = (
            float(np.mean(occupancy[rows[inside], cols[inside]] > 0))
            if np.any(inside)
            else 0.0
        )
        support_ratios.append(ratio)

        fitted: FloatArray | None = None
        if boundary.shape[0] >= 8:
            relative = boundary - start
            along_distance = relative @ direction
            perpendicular = np.abs(
                relative[:, 0] * direction[1] - relative[:, 1] * direction[0]
            )
            selected = boundary[
                (along_distance >= -2.0)
                & (along_distance <= length + 2.0)
                & (perpendicular <= max(3.0, 0.010 / resolution_m))
            ]
            if selected.shape[0] >= max(8, int(0.20 * length)):
                centered = selected - np.mean(selected, axis=0)
                _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
                if singular_values.size >= 2 and float(
                    singular_values[0]
                ) > 3.0 * float(singular_values[1]):
                    fitted = vh[0] / np.linalg.norm(vh[0])
        line_observed = fitted is not None
        observed.append(bool(ratio >= min_support_ratio and line_observed))
        fitted_directions.append(fitted)

    orthogonality_errors: list[float] = []
    for index in range(4):
        first = fitted_directions[index]
        second = fitted_directions[(index + 1) % 4]
        if first is None or second is None:
            continue
        angle = _line_angle_difference(first, second)
        orthogonality_errors.append(abs(0.5 * math.pi - angle))
    # Opposite corrugated-cardboard edges are often locally ragged at the
    # RealSense sampling scale.  The median adjacent-line error is the robust
    # global rectangle orthogonality statistic; observability still requires
    # at least three supported rims and both axis directions below.
    error = float(np.median(orthogonality_errors)) if orthogonality_errors else math.pi
    return tuple(support_ratios), tuple(observed), float(error)


def _sample_points(points: FloatArray, limit: int) -> FloatArray:
    if points.shape[0] <= limit:
        return points
    indices = np.linspace(0, points.shape[0] - 1, limit, dtype=np.int64)
    return points[indices]


def _fit_trimmed_line_2d(
    points_xy: FloatArray,
    *,
    expected_direction_xy: FloatArray,
) -> _LineFit2D:
    cloud = np.asarray(points_xy, dtype=np.float64)
    cloud = cloud[np.all(np.isfinite(cloud), axis=1)]
    if cloud.shape[0] < 12:
        raise ValueError("insufficient line support points")
    retained = cloud
    for _ in range(2):
        centroid = np.mean(retained, axis=0)
        _, singular_values, vh = np.linalg.svd(retained - centroid, full_matrices=False)
        if singular_values.size < 2 or float(singular_values[0]) <= 1e-9:
            raise ValueError("degenerate line support")
        direction = vh[0] / np.linalg.norm(vh[0])
        residuals = np.abs(
            (cloud[:, 0] - centroid[0]) * direction[1]
            - (cloud[:, 1] - centroid[1]) * direction[0]
        )
        cutoff = max(0.003, float(np.percentile(residuals, 85)))
        retained = cloud[residuals <= cutoff]
        if retained.shape[0] < 12:
            raise ValueError("line trimming removed too much support")
    centroid = np.mean(retained, axis=0)
    _, singular_values, vh = np.linalg.svd(retained - centroid, full_matrices=False)
    if (
        singular_values.size < 2
        or float(singular_values[0]) <= 1e-9
        or float(singular_values[0]) <= 2.0 * float(singular_values[1])
    ):
        raise ValueError("line support is not elongated")
    direction = vh[0] / np.linalg.norm(vh[0])
    expected = np.asarray(expected_direction_xy, dtype=np.float64)
    expected /= np.linalg.norm(expected)
    if float(direction @ expected) < 0.0:
        direction = -direction
    along = (retained - centroid) @ direction
    endpoints = np.stack(
        (
            centroid + float(np.min(along)) * direction,
            centroid + float(np.max(along)) * direction,
        )
    )
    residuals = np.abs(
        (retained[:, 0] - centroid[0]) * direction[1]
        - (retained[:, 1] - centroid[1]) * direction[0]
    )
    return _LineFit2D(
        centroid_xy=centroid,
        direction_xy=direction,
        endpoints_xy=endpoints,
        support_point_count=int(retained.shape[0]),
        support_length_m=float(np.ptp(along)),
        p95_residual_m=float(np.percentile(residuals, 95)),
        axis_residual_rad=_line_angle_difference(direction, expected),
    )


def _line_intersection_2d(first: _LineFit2D, second: _LineFit2D) -> FloatArray:
    system = np.column_stack((first.direction_xy, -second.direction_xy))
    determinant = float(np.linalg.det(system))
    if abs(determinant) <= math.sin(math.radians(10.0)):
        raise ValueError("boundary lines are parallel")
    parameters = np.linalg.solve(system, second.centroid_xy - first.centroid_xy)
    return first.centroid_xy + float(parameters[0]) * first.direction_xy


def _base_points_touch_image_crop(
    points_base: FloatArray,
    *,
    intrinsics: CameraIntrinsics,
    T_base_depth: FloatArray,
    margin_px: int,
) -> bool:
    points = np.asarray(points_base, dtype=np.float64)
    points_depth = (points - T_base_depth[:3, 3]) @ T_base_depth[:3, :3]
    z = points_depth[:, 2]
    if np.any(~np.isfinite(points_depth)) or np.any(z <= 1e-6):
        return True
    cols = float(intrinsics.fx) * points_depth[:, 0] / z + float(intrinsics.cx)
    rows = float(intrinsics.fy) * points_depth[:, 1] / z + float(intrinsics.cy)
    margin = float(margin_px)
    return bool(
        np.any(cols <= margin)
        or np.any(cols >= float(intrinsics.width - 1) - margin)
        or np.any(rows <= margin)
        or np.any(rows >= float(intrinsics.height - 1) - margin)
    )


def _points_touch_bev_crop(
    points_base: FloatArray,
    *,
    config: PalletEstimatorConfig,
) -> bool:
    points = np.asarray(points_base, dtype=np.float64)
    margin = config.l_corner_bev_crop_margin_m
    return bool(
        np.any(points[:, 0] <= config.workspace_x_m[0] + margin)
        or np.any(points[:, 0] >= config.workspace_x_m[1] - margin)
        or np.any(points[:, 1] <= config.workspace_y_m[0] + margin)
        or np.any(points[:, 1] >= config.workspace_y_m[1] - margin)
    )


def _line_evidence(
    fit: _LineFit2D,
    *,
    role: str,
    plane: Plane,
    intrinsics: CameraIntrinsics,
    T_base_depth: FloatArray,
    config: PalletEstimatorConfig,
) -> BoundaryLineEvidence:
    endpoints_z = _plane_z(plane, fit.endpoints_xy)
    endpoints_base = np.column_stack((fit.endpoints_xy, endpoints_z))
    direction_z = float(_plane_z(plane, fit.centroid_xy + fit.direction_xy)) - float(
        _plane_z(plane, fit.centroid_xy)
    )
    direction_base = _unit((fit.direction_xy[0], fit.direction_xy[1], direction_z))
    return BoundaryLineEvidence(
        role=role,
        endpoints_base=endpoints_base,
        direction_base=direction_base,
        support_length_m=fit.support_length_m,
        support_point_count=fit.support_point_count,
        p95_residual_m=fit.p95_residual_m,
        axis_residual_rad=fit.axis_residual_rad,
        touches_image_crop=_base_points_touch_image_crop(
            endpoints_base,
            intrinsics=intrinsics,
            T_base_depth=T_base_depth,
            margin_px=config.l_corner_image_crop_margin_px,
        ),
        touches_bev_crop=_points_touch_bev_crop(endpoints_base, config=config),
    )


def _held_carton_footprint_mask(
    points_base: FloatArray,
    workspace: NDArray[np.bool_],
    hint: HeldBoxHint | None,
) -> NDArray[np.bool_] | None:
    """Project the EEF-associated carton footprint once per frame."""

    if hint is None or hint.center_base is None:
        return None
    center = np.asarray(hint.center_base, dtype=np.float64)
    if hint.yaw_base_rad is None:
        return None
    yaw = float(hint.yaw_base_rad)
    axis_long = np.array((math.cos(yaw), math.sin(yaw)), dtype=np.float64)
    axis_short = np.array((-axis_long[1], axis_long[0]), dtype=np.float64)
    relative_xy = points_base[workspace, :2] - center[:2]
    along = relative_xy @ axis_long
    across = relative_xy @ axis_short
    footprint = hint.footprint_size_m
    footprint_mask = np.zeros(workspace.shape, dtype=np.bool_)
    footprint_mask[workspace] = (np.abs(along) <= 0.5 * footprint[0] + 0.08) & (
        np.abs(across) <= 0.5 * footprint[1] + 0.08
    )
    return footprint_mask


def _held_carton_exclusion_mask(
    points_base: FloatArray,
    workspace: NDArray[np.bool_],
    hint: HeldBoxHint | None,
    footprint_mask: NDArray[np.bool_] | None,
) -> NDArray[np.bool_]:
    """Return an EEF-associated exclusion only; it never selects a plane."""

    if hint is None or hint.eef_proxy_z_base_m is None or footprint_mask is None:
        return np.zeros(workspace.shape, dtype=np.bool_)
    # The EEF proxy is only an association datum.  The 120 mm lower band is
    # intentionally smaller than the measured carton height, so the lower
    # completed layer is not removed when the carried footprint overlaps it.
    return (
        workspace
        & footprint_mask
        & (points_base[..., 2] >= float(hint.eef_proxy_z_base_m) - 0.120)
    )


def _fit_partial_l_corner(
    occupancy: NDArray[np.uint8],
    *,
    stack_plane: _PlaneCandidate,
    intrinsics: CameraIntrinsics,
    T_base_depth: FloatArray,
    config: PalletEstimatorConfig,
    timestamp_s: float,
    calibration_status: str,
    held_excluded_point_count: int = 0,
) -> _LCornerFitResult:
    """Fit the near/image-right metric L without recovering stack scale."""

    cv2 = _cv2()
    plane_height = float(
        _plane_z(stack_plane.plane, np.array((0.0, 0.0), dtype=np.float64))
    )
    base_quality = {
        "stack_plane_p95_residual_m": stack_plane.p95_residual_m,
        "stack_plane_inlier_count": float(stack_plane.inlier_count),
        "held_selection_excluded_point_count": float(
            max(0, int(held_excluded_point_count))
        ),
    }
    all_unconstrained = (
        "stack_center_x",
        "stack_center_y",
        "hole_center_x",
        "hole_center_y",
        "slot1_target_x",
        "slot1_target_y",
    )

    def rejected(
        reasons: tuple[str, ...],
        *,
        constrained: tuple[str, ...] = ("stack_plane_z",),
        front_line: BoundaryLineEvidence | None = None,
        side_line: BoundaryLineEvidence | None = None,
        connection_gap_m: float | None = None,
        orthogonality_error_rad: float | None = None,
        branch: str | None = None,
        quality: dict[str, float] | None = None,
        component_boundary_base: FloatArray | None = None,
        forward_acquisition_valid: bool = False,
        forward_acquisition_yaw_base_rad: float | None = None,
        forward_acquisition_rejection_reasons: tuple[str, ...] = (),
    ) -> _LCornerFitResult:
        return _LCornerFitResult(
            observation=LCornerObservation(
                timestamp_s=timestamp_s,
                corner_base=None,
                u_right_base=None,
                v_far_base=None,
                yaw_base_rad=None,
                plane_height_base_m=plane_height,
                plane_p95_residual_m=stack_plane.p95_residual_m,
                front_line=front_line,
                side_line=side_line,
                connection_gap_m=connection_gap_m,
                orthogonality_error_rad=orthogonality_error_rad,
                topology_branch=branch,
                constrained_dofs=constrained,
                unconstrained_dofs=all_unconstrained,
                quality=base_quality if quality is None else quality,
                valid=False,
                rejection_reasons=reasons,
                forward_acquisition_valid=forward_acquisition_valid,
                forward_acquisition_yaw_base_rad=forward_acquisition_yaw_base_rad,
                forward_acquisition_rejection_reasons=(
                    forward_acquisition_rejection_reasons
                ),
                calibration_status=calibration_status,
            ),
            component_boundary_base=component_boundary_base,
        )

    component_count, component_labels, component_stats, _ = (
        cv2.connectedComponentsWithStats(
            occupancy,
            connectivity=8,
        )
    )
    if component_count <= 1:
        return rejected(("l_corner_component_missing",))
    component_label = max(
        range(1, component_count),
        key=lambda label: int(component_stats[label, cv2.CC_STAT_AREA]),
    )
    component = (component_labels == component_label).astype(np.uint8) * 255
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return rejected(("l_corner_component_missing",))
    boundary_grid = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    if boundary_grid.shape[0] < 24:
        return rejected(("l_corner_boundary_support_too_short",))
    boundary_xy = _grid_to_base_xy(
        boundary_grid,
        x_max=config.workspace_x_m[1],
        y_max=config.workspace_y_m[1],
        resolution_m=config.grid_resolution_m,
    )
    boundary_z = _plane_z(stack_plane.plane, boundary_xy)
    boundary_base = np.column_stack((boundary_xy, boundary_z))

    normal = np.asarray(stack_plane.plane.normal, dtype=np.float64)
    image_right = np.asarray(T_base_depth[:3, 0], dtype=np.float64)
    image_right = _unit(image_right - float(image_right @ normal) * normal)
    v_far = _unit(np.cross(normal, image_right))
    rough_front = np.array(
        (config.rough_front_axis_base[0], config.rough_front_axis_base[1], 0.0),
        dtype=np.float64,
    )
    rough_front = _unit(rough_front - float(rough_front @ normal) * normal)
    if float(v_far @ rough_front) <= 0.0:
        return rejected(
            ("l_corner_reflected_branch",),
            quality={
                **base_quality,
                "rough_front_signed_alignment": float(v_far @ rough_front),
            },
            component_boundary_base=boundary_base,
        )
    u_xy = image_right[:2] / np.linalg.norm(image_right[:2])
    v_xy = v_far[:2] / np.linalg.norm(v_far[:2])
    canonical = np.column_stack((boundary_xy @ u_xy, boundary_xy @ v_xy))
    edge_band = config.l_corner_edge_band_m
    near_level = float(np.percentile(canonical[:, 1], 3))
    front_points = boundary_xy[canonical[:, 1] <= near_level + edge_band]
    try:
        front_fit = _fit_trimmed_line_2d(
            front_points,
            expected_direction_xy=u_xy,
        )
    except ValueError:
        return rejected(
            ("l_corner_front_line_missing",),
            component_boundary_base=boundary_base,
        )
    front_line = _line_evidence(
        front_fit,
        role="near_front",
        plane=stack_plane.plane,
        intrinsics=intrinsics,
        T_base_depth=T_base_depth,
        config=config,
    )

    right_level = float(np.percentile(canonical[:, 0], 97))
    side_points = boundary_xy[
        (canonical[:, 0] >= right_level - edge_band)
        & (canonical[:, 1] >= near_level - 2.0 * edge_band)
    ]
    try:
        side_fit = _fit_trimmed_line_2d(
            side_points,
            expected_direction_xy=v_xy,
        )
    except ValueError:
        return rejected(
            ("l_corner_side_line_missing",),
            constrained=("stack_plane_z", "near_front_line"),
            front_line=front_line,
            component_boundary_base=boundary_base,
        )
    side_line = _line_evidence(
        side_fit,
        role="image_right_side",
        plane=stack_plane.plane,
        intrinsics=intrinsics,
        T_base_depth=T_base_depth,
        config=config,
    )

    reasons: list[str] = []
    if front_fit.support_length_m < config.l_corner_min_front_support_m:
        reasons.append("l_corner_front_support_too_short")
    if side_fit.support_length_m < config.l_corner_min_side_support_m:
        reasons.append("l_corner_side_support_too_short")
    if front_fit.p95_residual_m > config.l_corner_max_line_p95_residual_m:
        reasons.append("l_corner_front_residual_too_large")
    if side_fit.p95_residual_m > config.l_corner_max_line_p95_residual_m:
        reasons.append("l_corner_side_residual_too_large")
    if front_fit.axis_residual_rad > config.l_corner_max_axis_residual_rad:
        reasons.append("l_corner_front_axis_mismatch")
    if side_fit.axis_residual_rad > config.l_corner_max_axis_residual_rad:
        reasons.append("l_corner_side_axis_mismatch")
    if front_line.touches_bev_crop or side_line.touches_bev_crop:
        reasons.append("l_corner_boundary_bev_cropped")
    orthogonality = abs(
        0.5 * math.pi
        - math.acos(
            float(
                np.clip(
                    abs(front_fit.direction_xy @ side_fit.direction_xy),
                    0.0,
                    1.0,
                )
            )
        )
    )
    if orthogonality > config.l_corner_max_orthogonality_error_rad:
        reasons.append("l_corner_orthogonality_error")
    try:
        corner_xy = _line_intersection_2d(front_fit, side_fit)
    except ValueError:
        return rejected(
            tuple(dict.fromkeys((*reasons, "l_corner_lines_parallel"))),
            constrained=("stack_plane_z", "near_front_line", "image_right_side_line"),
            front_line=front_line,
            side_line=side_line,
            orthogonality_error_rad=orthogonality,
            branch="near_image_right_outer",
            forward_acquisition_rejection_reasons=(
                "l_corner_lines_parallel",
            ),
            component_boundary_base=boundary_base,
        )
    front_endpoint_distances = np.linalg.norm(
        front_fit.endpoints_xy - corner_xy, axis=1
    )
    side_endpoint_distances = np.linalg.norm(side_fit.endpoints_xy - corner_xy, axis=1)
    connection_gap = float(
        max(np.min(front_endpoint_distances), np.min(side_endpoint_distances))
    )
    if connection_gap > config.l_corner_max_connection_gap_m:
        reasons.append("l_corner_lines_disconnected")
    if (
        int(np.argmin(front_endpoint_distances)) != 1
        or int(np.argmin(side_endpoint_distances)) != 0
    ):
        reasons.append("l_corner_reflected_branch")

    front_direction_z = float(
        _plane_z(stack_plane.plane, front_fit.centroid_xy + front_fit.direction_xy)
    ) - float(_plane_z(stack_plane.plane, front_fit.centroid_xy))
    observed_u_right = _unit(
        (
            front_fit.direction_xy[0],
            front_fit.direction_xy[1],
            front_direction_z,
        )
    )
    observed_v_far = _unit(np.cross(normal, observed_u_right))
    if float(observed_v_far @ side_line.direction_base) <= 0.0:
        reasons.append("l_corner_reflected_branch")

    interior_xy = corner_xy - 0.025 * u_xy + 0.025 * v_xy
    interior_row = int(
        round((config.workspace_x_m[1] - interior_xy[0]) / config.grid_resolution_m)
    )
    interior_col = int(
        round((config.workspace_y_m[1] - interior_xy[1]) / config.grid_resolution_m)
    )
    if (
        interior_row < 0
        or interior_row >= component.shape[0]
        or interior_col < 0
        or interior_col >= component.shape[1]
        or component[interior_row, interior_col] == 0
    ):
        reasons.append("l_corner_topology_mismatch")
    if stack_plane.p95_residual_m > config.gates.max_plane_p95_residual_m:
        reasons.append("stack_plane_residual_too_large")

    branch = "near_image_right_outer"
    quality = {
        **base_quality,
        "l_corner_component_area_m2": float(
            component_stats[component_label, cv2.CC_STAT_AREA]
        )
        * config.grid_resolution_m**2,
        "l_corner_front_support_m": front_fit.support_length_m,
        "l_corner_side_support_m": side_fit.support_length_m,
        "l_corner_front_p95_residual_m": front_fit.p95_residual_m,
        "l_corner_side_p95_residual_m": side_fit.p95_residual_m,
        "l_corner_front_axis_residual_rad": front_fit.axis_residual_rad,
        "l_corner_side_axis_residual_rad": side_fit.axis_residual_rad,
        "l_corner_connection_gap_m": connection_gap,
        "l_corner_orthogonality_error_rad": orthogonality,
    }
    forward_acquisition_reasons: list[str] = []
    if front_fit.support_length_m < config.l_corner_acquisition_min_front_support_m:
        forward_acquisition_reasons.append("l_corner_acquisition_front_support_too_short")
    if side_fit.support_length_m < config.l_corner_acquisition_min_side_support_m:
        forward_acquisition_reasons.append("l_corner_acquisition_side_support_too_short")
    if front_fit.p95_residual_m > config.l_corner_acquisition_max_line_p95_residual_m:
        forward_acquisition_reasons.append(
            "l_corner_acquisition_front_residual_too_large"
        )
    if side_fit.p95_residual_m > config.l_corner_acquisition_max_line_p95_residual_m:
        forward_acquisition_reasons.append(
            "l_corner_acquisition_side_residual_too_large"
        )
    if front_fit.axis_residual_rad > config.l_corner_acquisition_max_axis_residual_rad:
        forward_acquisition_reasons.append("l_corner_acquisition_front_axis_mismatch")
    if side_fit.axis_residual_rad > config.l_corner_acquisition_max_axis_residual_rad:
        forward_acquisition_reasons.append("l_corner_acquisition_side_axis_mismatch")
    if orthogonality > config.l_corner_acquisition_max_orthogonality_error_rad:
        forward_acquisition_reasons.append("l_corner_acquisition_orthogonality_error")
    if connection_gap > config.l_corner_acquisition_max_connection_gap_m:
        forward_acquisition_reasons.append("l_corner_acquisition_gap_too_large")
    if stack_plane.p95_residual_m > config.gates.max_plane_p95_residual_m:
        forward_acquisition_reasons.append(
            "l_corner_acquisition_stack_plane_residual_too_large"
        )
    # The relaxed path authorizes only another bounded forward observe step,
    # but it must not turn crop/reflection/topology failures into evidence.
    # Only the deliberately relaxed support and connection requirements may
    # differ from the strict metric-corner contract.
    relaxed_strict_reasons = {
        "l_corner_front_support_too_short",
        "l_corner_side_support_too_short",
        "l_corner_lines_disconnected",
    }
    forward_acquisition_reasons.extend(
        reason for reason in reasons if reason not in relaxed_strict_reasons
    )
    forward_acquisition_yaw = float(
        math.atan2(observed_u_right[1], observed_u_right[0])
    )
    forward_acquisition_valid = not forward_acquisition_reasons
    if reasons:
        return rejected(
            tuple(dict.fromkeys(reasons)),
            constrained=("stack_plane_z", "near_front_line", "image_right_side_line"),
            front_line=front_line,
            side_line=side_line,
            connection_gap_m=connection_gap,
            orthogonality_error_rad=orthogonality,
            branch=branch,
            quality=quality,
            forward_acquisition_valid=forward_acquisition_valid,
            forward_acquisition_yaw_base_rad=forward_acquisition_yaw,
            forward_acquisition_rejection_reasons=tuple(
                dict.fromkeys(forward_acquisition_reasons)
            ),
            component_boundary_base=boundary_base,
        )

    corner_z = float(_plane_z(stack_plane.plane, corner_xy))
    corner_base = np.array((corner_xy[0], corner_xy[1], corner_z), dtype=np.float64)
    return _LCornerFitResult(
        observation=LCornerObservation(
            timestamp_s=timestamp_s,
            corner_base=corner_base,
            u_right_base=observed_u_right,
            v_far_base=observed_v_far,
            yaw_base_rad=forward_acquisition_yaw,
            plane_height_base_m=corner_z,
            plane_p95_residual_m=stack_plane.p95_residual_m,
            front_line=front_line,
            side_line=side_line,
            connection_gap_m=connection_gap,
            orthogonality_error_rad=orthogonality,
            topology_branch=branch,
            constrained_dofs=("corner_x", "corner_y", "stack_yaw", "stack_plane_z"),
            unconstrained_dofs=all_unconstrained,
            quality=quality,
            valid=True,
            rejection_reasons=(),
            forward_acquisition_valid=True,
            calibration_status=calibration_status,
        ),
        component_boundary_base=boundary_base,
    )


class PalletStackEstimator:
    """Hardware-independent stack/opening estimator with fail-closed outputs."""

    def __init__(self, config: PalletEstimatorConfig | None = None) -> None:
        self.config = PalletEstimatorConfig() if config is None else config
        self.last_evidence: PalletFrameEvidence | None = None
        self._ray_cache_key: tuple[Any, ...] | None = None
        self._base_ray_coefficients: FloatArray | None = None
        self._previous_valid_frame_id: int | None = None
        self._previous_valid_timestamp_s: float | None = None
        self._previous_valid_center: FloatArray | None = None

    def _ray_coefficients(
        self,
        intrinsics: CameraIntrinsics,
        T_base_depth: FloatArray,
    ) -> FloatArray:
        key = (
            intrinsics.width,
            intrinsics.height,
            intrinsics.fx,
            intrinsics.fy,
            intrinsics.cx,
            intrinsics.cy,
            *T_base_depth[:3, :].reshape(-1).tolist(),
        )
        if key == self._ray_cache_key and self._base_ray_coefficients is not None:
            return self._base_ray_coefficients
        rows, cols = np.indices((intrinsics.height, intrinsics.width), dtype=np.float64)
        ray_x = (cols - float(intrinsics.cx)) / float(intrinsics.fx)
        ray_y = (rows - float(intrinsics.cy)) / float(intrinsics.fy)
        coefficients = (
            ray_x[..., None] * T_base_depth[:3, 0]
            + ray_y[..., None] * T_base_depth[:3, 1]
            + T_base_depth[:3, 2]
        )
        coefficients.setflags(write=False)
        self._ray_cache_key = key
        self._base_ray_coefficients = coefficients
        return coefficients

    def _failure(
        self,
        reason: str,
        *,
        timestamp_s: float,
        calibration_status: str,
        quality: dict[str, float] | None = None,
        coarse: LCornerObservation | None = None,
        evidence: PalletFrameEvidence | None = None,
    ) -> PalletSceneObservation:
        self.last_evidence = evidence
        # Preserve the most recent *valid* center across dropped/invalid frames.
        # The timestamp-age gate decides whether it is still recent enough for
        # the next valid observation to be jump-checked.
        return PalletSceneObservation(
            stack=StackObservation(
                timestamp_s=timestamp_s,
                center_base=None,
                u_right_base=None,
                v_far_base=None,
                yaw_base_rad=None,
                plane_height_base_m=None,
                slot1_target_base=None,
                opening_size_m=None,
                quality={} if quality is None else quality,
                valid=False,
                rejection_reasons=(reason,),
                calibration_status=calibration_status,
            ),
            held_top=None,
            coarse=coarse,
        )

    def estimate(
        self,
        depth_m: ArrayLike,
        depth_intrinsics: CameraIntrinsics,
        T_base_depth: ArrayLike,
        *,
        timestamp_s: float = 0.0,
        frame_id: int = 0,
        color_on_depth_bgr: ArrayLike | None = None,
        held_box_hint: HeldBoxHint | None = None,
        calibration_status: str = "nominal_ready_assumed",
    ) -> PalletSceneObservation:
        """Estimate one stack observation from metric depth.

        ``frame_id`` is accepted for logging symmetry; geometry is stateless and
        never median-filters frames collected while the base may be moving.
        """

        self.last_evidence = None
        frame_number = int(frame_id)
        timestamp = float(timestamp_s)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp_s must be finite")
        if not isinstance(depth_intrinsics, CameraIntrinsics):
            return self._failure(
                "invalid_depth_or_metadata",
                timestamp_s=timestamp,
                calibration_status=calibration_status,
            )
        try:
            transform = validate_transform(T_base_depth, name="T_base_depth")
        except ValueError:
            return self._failure(
                "invalid_base_depth_transform",
                timestamp_s=timestamp,
                calibration_status=calibration_status,
            )
        depth = np.asarray(depth_m)
        expected_shape = (depth_intrinsics.height, depth_intrinsics.width)
        if (
            depth.shape != expected_shape
            or np.issubdtype(depth.dtype, np.integer)
            or not np.issubdtype(depth.dtype, np.floating)
        ):
            return self._failure(
                "invalid_metric_depth",
                timestamp_s=timestamp,
                calibration_status=calibration_status,
            )
        depth = depth.astype(np.float64, copy=False)
        if color_on_depth_bgr is not None:
            color = np.asarray(color_on_depth_bgr)
            if color.dtype != np.uint8 or color.shape != (*expected_shape, 3):
                return self._failure(
                    "invalid_aligned_color",
                    timestamp_s=timestamp,
                    calibration_status=calibration_status,
                )
        config = self.config
        coefficients = self._ray_coefficients(depth_intrinsics, transform)
        points_base = depth[..., None] * coefficients + transform[:3, 3]
        # Intrinsics and ``transform`` are finite-validated.  Any overflowed
        # base coordinate still fails the paired workspace bounds below, so a
        # second three-channel finiteness reduction cannot admit another point.
        finite = np.isfinite(depth)
        workspace = (
            finite
            & (depth >= config.min_depth_m)
            & (depth <= config.max_depth_m)
            & (points_base[..., 0] >= config.workspace_x_m[0])
            & (points_base[..., 0] <= config.workspace_x_m[1])
            & (points_base[..., 1] >= config.workspace_y_m[0])
            & (points_base[..., 1] <= config.workspace_y_m[1])
            & (points_base[..., 2] >= config.workspace_z_m[0])
            & (points_base[..., 2] <= config.workspace_z_m[1])
        )
        workspace_count = int(np.count_nonzero(workspace))
        if workspace_count < config.min_plane_points:
            return self._failure(
                "insufficient_workspace_points",
                timestamp_s=timestamp,
                calibration_status=calibration_status,
                quality={"workspace_point_count": float(workspace_count)},
            )

        held_footprint_mask = _held_carton_footprint_mask(
            points_base,
            workspace,
            held_box_hint,
        )
        held_selection_exclusion = _held_carton_exclusion_mask(
            points_base,
            workspace,
            held_box_hint,
            held_footprint_mask,
        )
        plane_selection_workspace = workspace & ~held_selection_exclusion
        held_selection_excluded_count = int(np.count_nonzero(held_selection_exclusion))

        try:
            peak_z, peak_count = _dominant_height(
                points_base[..., 2][plane_selection_workspace],
                lower_m=config.stack_plane_z_m[0],
                upper_m=config.stack_plane_z_m[1],
                bin_m=config.z_histogram_bin_m,
            )
        except ValueError:
            return self._failure(
                "stack_plane_missing",
                timestamp_s=timestamp,
                calibration_status=calibration_status,
                quality={"workspace_point_count": float(workspace_count)},
            )
        seed_mask = plane_selection_workspace & (
            np.abs(points_base[..., 2] - peak_z) <= config.plane_seed_band_m
        )
        seed_points = points_base[seed_mask]
        if seed_points.shape[0] < config.min_plane_points:
            return self._failure(
                "insufficient_stack_plane_points",
                timestamp_s=timestamp,
                calibration_status=calibration_status,
                quality={
                    "workspace_point_count": float(workspace_count),
                    "stack_height_peak_point_count": float(peak_count),
                },
            )
        try:
            stack_plane = _fit_trimmed_horizontal_plane(
                seed_points,
                slab_m=config.plane_fit_tolerance_m,
                max_points=config.plane_fit_max_points,
            )
        except ValueError:
            return self._failure(
                "stack_plane_fit_failed",
                timestamp_s=timestamp,
                calibration_status=calibration_status,
                quality={"stack_plane_seed_count": float(seed_points.shape[0])},
            )
        normal = np.asarray(stack_plane.plane.normal, dtype=np.float64)
        if float(normal[2]) < math.cos(math.radians(10.0)):
            return self._failure(
                "stack_plane_not_horizontal",
                timestamp_s=timestamp,
                calibration_status=calibration_status,
                quality={"stack_plane_normal_z": float(normal[2])},
            )

        signed_height = points_base @ normal - float(stack_plane.plane.d)
        stack_mask = (
            workspace
            & ~held_selection_exclusion
            & (np.abs(signed_height) <= config.plane_slab_m)
        )
        stack_points = points_base[stack_mask]
        if stack_points.shape[0] < config.min_plane_points:
            return self._failure(
                "insufficient_stack_plane_points",
                timestamp_s=timestamp,
                calibration_status=calibration_status,
                quality={"stack_plane_point_count": float(stack_points.shape[0])},
            )

        cv2 = _cv2()
        resolution = config.grid_resolution_m
        x_min, x_max = config.workspace_x_m
        y_min, y_max = config.workspace_y_m
        grid_height = int(math.ceil((x_max - x_min) / resolution)) + 1
        grid_width = int(math.ceil((y_max - y_min) / resolution)) + 1
        rows = np.clip(
            ((x_max - stack_points[:, 0]) / resolution).astype(np.int64),
            0,
            grid_height - 1,
        )
        cols = np.clip(
            ((y_max - stack_points[:, 1]) / resolution).astype(np.int64),
            0,
            grid_width - 1,
        )
        occupancy = np.zeros((grid_height, grid_width), dtype=np.uint8)
        occupancy[rows, cols] = 255
        close_size = _kernel_size(config.morphology_close_m, resolution)
        dilate_size = _kernel_size(config.morphology_dilate_m, resolution)
        occupancy = cv2.morphologyEx(
            occupancy,
            cv2.MORPH_CLOSE,
            np.ones((close_size, close_size), dtype=np.uint8),
        )
        occupancy = cv2.dilate(
            occupancy,
            np.ones((dilate_size, dilate_size), dtype=np.uint8),
        )
        contours, _ = cv2.findContours(
            occupancy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return self._failure(
                "stack_top_component_missing",
                timestamp_s=timestamp,
                calibration_status=calibration_status,
            )
        hull_points = cv2.convexHull(np.vstack(contours))
        hull = np.zeros_like(occupancy)
        cv2.fillConvexPoly(hull, hull_points, 255)
        l_corner_fit = _fit_partial_l_corner(
            occupancy,
            stack_plane=stack_plane,
            intrinsics=depth_intrinsics,
            T_base_depth=transform,
            config=config,
            timestamp_s=timestamp,
            calibration_status=calibration_status,
            held_excluded_point_count=held_selection_excluded_count,
        )
        coarse = l_corner_fit.observation
        closer_mask = workspace & (
            signed_height >= 0.5 * config.held_plane_min_separation_m
        )
        closer_rejected = int(np.count_nonzero(closer_mask))
        coarse_evidence = PalletFrameEvidence(
            stack_plane_base=stack_plane.plane,
            stack_points_base=_sample_points(stack_points, 2_500),
            closer_points_rejected=closer_rejected,
            held_excluded_points_base=_sample_points(
                points_base[held_selection_exclusion],
                1_000,
            ),
            l_corner_component_points_base=(
                None
                if l_corner_fit.component_boundary_base is None
                else _sample_points(l_corner_fit.component_boundary_base, 1_500)
            ),
            l_corner_front_endpoints_base=(
                None if coarse.front_line is None else coarse.front_line.endpoints_base
            ),
            l_corner_side_endpoints_base=(
                None if coarse.side_line is None else coarse.side_line.endpoints_base
            ),
            l_corner_corner_base=coarse.corner_base,
        )
        holes = cv2.bitwise_and(hull, cv2.bitwise_not(occupancy))
        holes = cv2.morphologyEx(
            holes,
            cv2.MORPH_OPEN,
            np.ones((3, 3), dtype=np.uint8),
        )
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            holes, connectivity=8
        )
        del component_count
        candidates = _opening_candidates(
            labels,
            stats,
            resolution_m=resolution,
            expected_m=config.geometry.opening_size_m,
            minimum_m=config.opening_component_min_m,
            maximum_m=config.opening_component_max_m,
        )
        if not candidates:
            return self._failure(
                "inner_opening_not_found",
                timestamp_s=timestamp,
                calibration_status=calibration_status,
                quality={
                    "stack_plane_p95_residual_m": stack_plane.p95_residual_m,
                    "stack_plane_point_count": float(stack_points.shape[0]),
                    "held_selection_excluded_point_count": float(
                        held_selection_excluded_count
                    ),
                },
                coarse=coarse,
                evidence=coarse_evidence,
            )
        opening = candidates[0]
        component = (labels == opening.label).astype(np.uint8) * 255
        rim_ratios, rim_observed, orthogonality_error = _side_evidence(
            occupancy,
            component,
            opening.box_grid,
            resolution_m=resolution,
            outer_band_m=config.rim_outer_band_m,
            min_support_ratio=config.min_rim_support_ratio,
        )

        corners_xy = _grid_to_base_xy(
            opening.box_grid,
            x_max=x_max,
            y_max=y_max,
            resolution_m=resolution,
        )
        center_xy = np.mean(corners_xy, axis=0)
        corner_z = _plane_z(stack_plane.plane, corners_xy)
        corners_base = np.column_stack((corners_xy, corner_z))
        center_base = np.array(
            (center_xy[0], center_xy[1], float(_plane_z(stack_plane.plane, center_xy))),
            dtype=np.float64,
        )

        edge_vectors = np.roll(corners_base, -1, axis=0) - corners_base
        edge_lengths = np.linalg.norm(edge_vectors, axis=1)
        edge_directions = edge_vectors / np.maximum(edge_lengths[:, None], 1e-12)
        image_right = np.asarray(transform[:3, 0], dtype=np.float64)
        image_right = image_right - float(image_right @ normal) * normal
        image_right = _unit(image_right)
        u_index = int(np.argmax(np.abs(edge_directions @ image_right)))
        u_right = edge_directions[u_index]
        if float(u_right @ image_right) < 0.0:
            u_right = -u_right
        u_right = _unit(u_right - float(u_right @ normal) * normal)
        v_far = _unit(np.cross(normal, u_right))
        rough_front = np.array(
            (config.rough_front_axis_base[0], config.rough_front_axis_base[1], 0.0),
            dtype=np.float64,
        )
        rough_front = _unit(rough_front - float(rough_front @ normal) * normal)
        start_yaw_residual = _line_angle_difference(v_far, rough_front)

        u_opening_size = float(edge_lengths[u_index])
        v_opening_size = float(edge_lengths[(u_index + 1) % 4])
        measured_opening = (u_opening_size, v_opening_size)
        expected_opening = config.geometry.opening_size_m
        axis_size_errors = (
            abs(measured_opening[0] - expected_opening[0]),
            abs(measured_opening[1] - expected_opening[1]),
        )
        slot_offset = config.geometry.slot1_offset_m
        slot_target = center_base + slot_offset[0] * u_right + slot_offset[1] * v_far
        yaw = float(math.atan2(u_right[1], u_right[0]))

        outer_rectangle = cv2.minAreaRect(hull_points.reshape(-1, 2).astype(np.float32))
        outer_size = tuple(
            float(value)
            for value in np.sort(
                np.asarray(outer_rectangle[1], dtype=np.float64) * resolution
            )
        )
        expected_outer = np.sort(
            np.asarray(config.geometry.outer_size_m, dtype=np.float64)
        )
        outer_error = float(np.linalg.norm(np.asarray(outer_size) - expected_outer))
        outer_observed = bool(outer_error <= 0.10)

        rim_count = int(sum(rim_observed))
        rim_direction_counts = (
            int(rim_observed[u_index]) + int(rim_observed[(u_index + 2) % 4]),
            int(rim_observed[(u_index + 1) % 4]) + int(rim_observed[(u_index + 3) % 4]),
        )
        rejection_reasons: list[str] = []
        if max(axis_size_errors) > config.gates.max_opening_size_error_m:
            rejection_reasons.append("opening_size_mismatch")
        if rim_count < config.gates.min_inner_rim_count:
            rejection_reasons.append("insufficient_inner_rims")
        if min(rim_direction_counts) < 1:
            rejection_reasons.append("inner_rims_single_direction")
        if orthogonality_error > config.gates.max_orthogonality_error_rad:
            rejection_reasons.append("opening_orthogonality_error")
        if stack_plane.p95_residual_m > config.gates.max_plane_p95_residual_m:
            rejection_reasons.append("stack_plane_residual_too_large")
        if start_yaw_residual > config.gates.max_start_yaw_residual_rad:
            rejection_reasons.append("rough_front_yaw_exceeded")
        if (
            self._previous_valid_frame_id is not None
            and self._previous_valid_timestamp_s is not None
            and self._previous_valid_center is not None
        ):
            center_jump_age_s = float(timestamp_s) - float(
                self._previous_valid_timestamp_s
            )
            if center_jump_age_s < -1e-6:
                rejection_reasons.append("center_jump_timestamp_regressed")
            elif center_jump_age_s <= config.gates.max_consecutive_center_jump_age_s:
                center_jump_m = float(
                    np.linalg.norm(center_base[:2] - self._previous_valid_center[:2])
                )
                if center_jump_m > config.gates.max_consecutive_center_jump_m:
                    rejection_reasons.append("center_jump_exceeded")

        held_observation: HeldBoxTopObservation | None = None
        # Every closer point is excluded from stack fitting.  A particular
        # closer plane is labelled "held box" only when fresh EEF geometry
        # supplies an association region; replay without robot state must not
        # confidently relabel a robot link or the box underside as its top.
        held_evidence = points_base[closer_mask]
        try:
            if held_box_hint is None or held_box_hint.center_base is None:
                raise ValueError("held box association requires an EEF center hint")
            if held_footprint_mask is None:
                raise ValueError("held box association requires an EEF footprint")
            association = closer_mask & held_footprint_mask
            lower_held_z = float(center_base[2] + config.held_plane_min_separation_m)
            held_peak, _ = _dominant_height(
                points_base[..., 2][association],
                lower_m=lower_held_z,
                upper_m=config.workspace_z_m[1],
                bin_m=config.z_histogram_bin_m,
            )
            held_seed_mask = association & (
                np.abs(points_base[..., 2] - held_peak) <= config.plane_seed_band_m
            )
            held_plane = _fit_trimmed_horizontal_plane(
                points_base[held_seed_mask],
                slab_m=config.plane_fit_tolerance_m,
                max_points=config.plane_fit_max_points,
            )
            held_normal = np.asarray(held_plane.plane.normal, dtype=np.float64)
            held_mask = association & (
                np.abs(points_base @ held_normal - float(held_plane.plane.d))
                <= config.plane_slab_m
            )
            held_evidence = points_base[held_mask]
            held_z = float(np.median(held_evidence[:, 2]))
            distinct = bool(
                held_z - center_base[2] >= config.held_plane_min_separation_m
            )
            if held_evidence.shape[0] >= 100:
                footprint = tuple(
                    float(value)
                    for value in np.sort(
                        np.percentile(held_evidence[:, :2], 97.5, axis=0)
                        - np.percentile(held_evidence[:, :2], 2.5, axis=0)
                    )
                )
            else:
                footprint = None
            eef_z = held_box_hint.eef_proxy_z_base_m
            delta_z = None if eef_z is None else held_z - float(eef_z)
            held_reasons: list[str] = []
            if not distinct:
                held_reasons.append("held_plane_not_distinct_from_stack")
            if held_plane.p95_residual_m > config.held_plane_max_uncertainty_m:
                held_reasons.append("held_plane_residual_too_large")
            held_observation = HeldBoxTopObservation(
                timestamp_s=timestamp,
                top_plane_z_base_m=held_z,
                top_plane_z_uncertainty_m=held_plane.p95_residual_m,
                eef_proxy_z_base_m=eef_z,
                delta_z_top_eef_m=delta_z,
                footprint_size_m=footprint,
                distinct_from_stack=distinct,
                valid=not held_reasons,
                rejection_reasons=tuple(held_reasons),
            )
        except ValueError:
            if held_box_hint is not None:
                held_observation = HeldBoxTopObservation(
                    timestamp_s=timestamp,
                    top_plane_z_base_m=None,
                    top_plane_z_uncertainty_m=None,
                    eef_proxy_z_base_m=held_box_hint.eef_proxy_z_base_m,
                    delta_z_top_eef_m=None,
                    footprint_size_m=held_box_hint.footprint_size_m,
                    distinct_from_stack=False,
                    valid=False,
                    rejection_reasons=("held_top_plane_unobservable",),
                )

        quality = {
            "workspace_point_count": float(workspace_count),
            "stack_height_peak_m": float(peak_z),
            "stack_height_peak_point_count": float(peak_count),
            "stack_plane_seed_count": float(stack_plane.seed_count),
            "stack_plane_inlier_count": float(stack_plane.inlier_count),
            "stack_plane_p95_residual_m": stack_plane.p95_residual_m,
            "stack_plane_normal_z": float(normal[2]),
            "inner_opening_candidate_count": float(len(candidates)),
            "inner_rim_count": float(rim_count),
            "opening_u_m": measured_opening[0],
            "opening_v_m": measured_opening[1],
            "opening_u_error_m": axis_size_errors[0],
            "opening_v_error_m": axis_size_errors[1],
            "orthogonality_error_rad": float(orthogonality_error),
            "rough_front_yaw_residual_rad": float(start_yaw_residual),
            "outer_observed": float(outer_observed),
            "outer_size_error_norm_m": outer_error,
            "closer_points_rejected": float(closer_rejected),
            "held_selection_excluded_point_count": float(held_selection_excluded_count),
        }
        valid = not rejection_reasons
        observation = PalletSceneObservation(
            stack=StackObservation(
                timestamp_s=timestamp,
                center_base=center_base,
                u_right_base=u_right,
                v_far_base=v_far,
                yaw_base_rad=yaw,
                plane_height_base_m=float(center_base[2]),
                slot1_target_base=slot_target,
                opening_size_m=measured_opening,
                quality=quality,
                valid=valid,
                rejection_reasons=tuple(rejection_reasons),
                calibration_status=calibration_status,
                axis_branch="image_right",
            ),
            held_top=held_observation,
            coarse=coarse,
        )
        self.last_evidence = PalletFrameEvidence(
            stack_plane_base=stack_plane.plane,
            opening_corners_base=corners_base,
            stack_points_base=_sample_points(stack_points, 2_500),
            held_points_base=_sample_points(held_evidence, 1_000),
            rim_support_ratios=rim_ratios,
            rim_observed=rim_observed,
            outer_size_m=outer_size if outer_observed else None,
            closer_points_rejected=closer_rejected,
            held_excluded_points_base=_sample_points(
                points_base[held_selection_exclusion],
                1_000,
            ),
            l_corner_component_points_base=(
                None
                if l_corner_fit.component_boundary_base is None
                else _sample_points(l_corner_fit.component_boundary_base, 1_500)
            ),
            l_corner_front_endpoints_base=(
                None if coarse.front_line is None else coarse.front_line.endpoints_base
            ),
            l_corner_side_endpoints_base=(
                None if coarse.side_line is None else coarse.side_line.endpoints_base
            ),
            l_corner_corner_base=coarse.corner_base,
        )
        if valid:
            self._previous_valid_frame_id = frame_number
            self._previous_valid_timestamp_s = float(timestamp_s)
            self._previous_valid_center = center_base.copy()
        return observation


__all__ = ["PalletStackEstimator"]
