"""Deterministic robust fitting of a fixed metric parcel rectangle.

The fit never optimizes physical scale.  It searches only center and line yaw
in continuous metric plane coordinates.  Image-border evidence, when
available, is treated as censoring rather than as a physical box edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .models import BoxModel


FloatArray = NDArray[np.float64]


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True)
class RectangleFitConfig:
    min_points: int = 80
    max_fit_points: int = 12_000
    coarse_angle_step_deg: float = 2.0
    fine_angle_step_deg: float = 0.10
    fine_half_width_deg: float = 2.2
    containment_tolerance_m: float = 0.010
    edge_band_m: float = 0.012
    robust_quantile: float = 0.004
    border_margin_px: int = 6
    censored_fraction: float = 0.18
    full_extent_tolerance_m: float = 0.025
    min_side_span_ratio: float = 0.30
    min_axis_assignment_margin: float = 0.045
    min_yaw_curvature: float = 5e-5

    def __post_init__(self) -> None:
        if self.min_points < 3 or self.max_fit_points < self.min_points:
            raise ValueError("rectangle point limits are inconsistent")
        if self.coarse_angle_step_deg <= 0.0 or self.fine_angle_step_deg <= 0.0:
            raise ValueError("angle search steps must be positive")
        if not 0.0 <= self.robust_quantile < 0.1:
            raise ValueError("robust_quantile must be in [0, 0.1)")


@dataclass(frozen=True)
class AxisFeasibleInterval:
    axis: str
    lower_m: float
    upper_m: float

    def to_dict(self) -> dict[str, Any]:
        return {"axis": self.axis, "parameter_interval_m": [self.lower_m, self.upper_m]}


@dataclass(frozen=True)
class RectangleFitResult:
    center_xy_m: tuple[float, float] | None
    yaw_rad: float | None
    long_axis_xy: tuple[float, float] | None
    short_axis_xy: tuple[float, float] | None
    corners_xy_m: tuple[tuple[float, float], ...] | None
    observability: dict[str, str]
    feasible_intervals: tuple[AxisFeasibleInterval, ...]
    side_support: dict[str, dict[str, Any]]
    score: float
    second_score: float
    candidate_margin: float
    yaw_curvature: float
    retained_fraction: float
    reasons: tuple[str, ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def valid_yaw(self) -> bool:
        return self.yaw_rad is not None

    @property
    def valid_center(self) -> bool:
        return self.center_xy_m is not None

    @property
    def full_pose_valid(self) -> bool:
        return self.valid_yaw and self.valid_center

    @property
    def geometry_valid(self) -> bool:
        return self.valid_yaw or self.valid_center

    @property
    def feasible_set(self) -> dict[str, list[float]] | None:
        if not self.feasible_intervals:
            return None
        return {
            item.axis: [item.lower_m, item.upper_m]
            for item in self.feasible_intervals
        }

    def to_dict(self) -> dict[str, Any]:
        return _json_value(
            {
                "center_xy_m": self.center_xy_m,
                "yaw_rad": self.yaw_rad,
                "long_axis_xy": self.long_axis_xy,
                "short_axis_xy": self.short_axis_xy,
                "corners_xy_m": self.corners_xy_m,
                "observability": self.observability,
                "feasible_intervals": [item.to_dict() for item in self.feasible_intervals],
                "side_support": self.side_support,
                "score": self.score,
                "second_score": self.second_score,
                "candidate_margin": self.candidate_margin,
                "yaw_curvature": self.yaw_curvature,
                "retained_fraction": self.retained_fraction,
                "reasons": self.reasons,
                "diagnostics": self.diagnostics,
            }
        )


@dataclass(frozen=True)
class _Side:
    coordinate_m: float
    support_points: int
    span_ratio: float
    near_border_fraction: float
    censored: bool
    physical: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "coordinate_m": self.coordinate_m,
            "support_points": self.support_points,
            "span_ratio": self.span_ratio,
            "near_border_fraction": self.near_border_fraction,
            "censored": self.censored,
            "physical": self.physical,
        }


@dataclass(frozen=True)
class _Axis:
    status: str
    center_m: float
    feasible_lower_m: float
    feasible_upper_m: float
    low: _Side
    high: _Side
    observed_span_m: float


@dataclass(frozen=True)
class _Candidate:
    theta: float
    score: float
    retained_fraction: float
    retained_points: int
    long_axis: _Axis
    short_axis: _Axis
    edge_quality: float


def _line_normalize(theta: float) -> float:
    return float(theta % math.pi)


def _validate_points(points_xy: ArrayLike) -> FloatArray:
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points_xy must have shape (N, 2), got {points.shape}")
    return points[np.all(np.isfinite(points), axis=1)]


def _even_sample(points: FloatArray, pixels: FloatArray | None, limit: int) -> tuple[FloatArray, FloatArray | None]:
    if points.shape[0] <= limit:
        return points, pixels
    indices = np.linspace(0, points.shape[0] - 1, limit, dtype=np.int64)
    return points[indices], None if pixels is None else pixels[indices]


def _densest_fixed_window(values: FloatArray, width: float) -> tuple[NDArray[np.bool_], float, float]:
    """Select the deterministic maximum-cardinality 1D fixed-width window."""

    # Candidate scoring calls this twice for every searched angle. Sorting
    # scalar values directly avoids building and gathering through an index
    # permutation. Equal-value source order cannot change the chosen interval.
    sorted_values = np.sort(values, kind="quicksort")
    right = np.searchsorted(sorted_values, sorted_values + width, side="right")
    counts = right - np.arange(sorted_values.size)
    start_index = int(np.argmax(counts))
    stop_index = int(right[start_index])
    low = float(sorted_values[start_index])
    high = low + float(width)
    mask = (values >= low) & (values <= high)
    # Guard against floating point disagreement with searchsorted.
    if int(np.count_nonzero(mask)) < stop_index - start_index:
        order = np.argsort(values, kind="mergesort")
        chosen = order[start_index:stop_index]
        mask[chosen] = True
    return mask, low, high


def _robust_bounds(values: FloatArray, quantile: float) -> tuple[float, float]:
    if values.size == 0:
        return math.nan, math.nan
    if values.size < 20 or quantile <= 0.0:
        return float(np.min(values)), float(np.max(values))
    low, high = _linear_quantile_pair(values, quantile, 1.0 - quantile)
    return float(low), float(high)


def _linear_quantile_pair(
    values: FloatArray,
    lower_quantile: float,
    upper_quantile: float,
) -> tuple[float, float]:
    """Return two NumPy-linear quantiles with one order-statistic partition.

    Candidate fitting only needs paired quantiles from finite one-dimensional
    arrays.  Using the four surrounding order statistics directly avoids the
    substantial generic ``np.quantile`` dispatch/allocation cost in the inner
    angle-search loop while preserving its ``method="linear"`` definition.
    """

    count = int(values.size)
    if count == 0:
        return math.nan, math.nan

    lower_index = float(lower_quantile) * (count - 1)
    upper_index = float(upper_quantile) * (count - 1)
    lower_floor = int(math.floor(lower_index))
    lower_ceil = int(math.ceil(lower_index))
    upper_floor = int(math.floor(upper_index))
    upper_ceil = int(math.ceil(upper_index))
    order = np.partition(
        values,
        (lower_floor, lower_ceil, upper_floor, upper_ceil),
    )

    def interpolate(index: float, floor_index: int, ceil_index: int) -> float:
        lower = float(order[floor_index])
        if floor_index == ceil_index:
            return lower
        upper = float(order[ceil_index])
        fraction = index - floor_index
        difference = upper - lower
        # Match NumPy's stable _lerp evaluation order on both sides of the
        # midpoint so this specialization is bit-for-bit equivalent too.
        if fraction >= 0.5:
            return upper - difference * (1.0 - fraction)
        return lower + difference * fraction

    return (
        interpolate(lower_index, lower_floor, lower_ceil),
        interpolate(upper_index, upper_floor, upper_ceil),
    )


def _near_border(pixels: FloatArray | None, image_shape: tuple[int, int] | None, margin: int) -> NDArray[np.bool_] | None:
    if pixels is None or image_shape is None:
        return None
    height, width = int(image_shape[0]), int(image_shape[1])
    u = pixels[:, 0]
    v = pixels[:, 1]
    return (u <= margin) | (u >= width - 1 - margin) | (v <= margin) | (v >= height - 1 - margin)


def _side(
    values: FloatArray,
    cross_values: FloatArray,
    coordinate: float,
    cross_length: float,
    near_border: NDArray[np.bool_] | None,
    *,
    config: RectangleFitConfig,
) -> _Side:
    selection = np.abs(values - coordinate) <= config.edge_band_m
    count = int(np.count_nonzero(selection))
    if count >= 4:
        lo, hi = _linear_quantile_pair(cross_values[selection], 0.03, 0.97)
        span_ratio = float(np.clip((hi - lo) / max(cross_length, 1e-9), 0.0, 1.5))
    else:
        span_ratio = 0.0
    border_fraction = (
        float(np.mean(near_border[selection]))
        if near_border is not None and count > 0
        else 0.0
    )
    censored = near_border is not None and border_fraction >= config.censored_fraction
    physical = not censored and span_ratio >= config.min_side_span_ratio
    return _Side(
        coordinate_m=float(coordinate),
        support_points=count,
        span_ratio=span_ratio,
        near_border_fraction=border_fraction,
        censored=censored,
        physical=physical,
    )


def _axis_observation(
    values: FloatArray,
    cross_values: FloatArray,
    length: float,
    cross_length: float,
    near_border: NDArray[np.bool_] | None,
    *,
    config: RectangleFitConfig,
) -> _Axis:
    low_value, high_value = _robust_bounds(values, config.robust_quantile)
    observed_span = max(0.0, high_value - low_value)
    low = _side(values, cross_values, low_value, cross_length, near_border, config=config)
    high = _side(values, cross_values, high_value, cross_length, near_border, config=config)

    feasible_low = high_value - 0.5 * length - config.containment_tolerance_m
    feasible_high = low_value + 0.5 * length + config.containment_tolerance_m
    if feasible_low > feasible_high:
        midpoint = 0.5 * (feasible_low + feasible_high)
        feasible_low = feasible_high = midpoint

    extent_complete = abs(observed_span - length) <= config.full_extent_tolerance_m
    if extent_complete and not (low.censored or high.censored):
        status = "both_edges"
        center = 0.5 * (low_value + high_value)
    elif low.physical and high.censored:
        status = "one_edge_inferred"
        center = low_value + 0.5 * length
        feasible_low = feasible_high = center
    elif low.censored and high.physical:
        status = "one_edge_inferred"
        center = high_value - 0.5 * length
        feasible_low = feasible_high = center
    elif low.physical and high.physical:
        # Two resolved physical edges are stronger than a small extent error
        # caused by stereo boundary noise or the robust trim.
        status = "both_edges"
        center = 0.5 * (low_value + high_value)
    else:
        status = "underconstrained"
        center = 0.5 * (feasible_low + feasible_high)
    return _Axis(
        status=status,
        center_m=float(center),
        feasible_lower_m=float(feasible_low),
        feasible_upper_m=float(feasible_high),
        low=low,
        high=high,
        observed_span_m=float(observed_span),
    )


def _candidate(
    points: FloatArray,
    near_border: NDArray[np.bool_] | None,
    theta: float,
    long_m: float,
    short_m: float,
    config: RectangleFitConfig,
) -> _Candidate:
    # Every scalar projection and every reconstructed axis must use the same
    # representative of the unoriented line.  Fine-search offsets can cross
    # 0/pi; normalizing only when storing the candidate flips both axes while
    # leaving the scalar centers in the old basis and mirrors the XY center.
    theta = _line_normalize(theta)
    long_dir = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
    short_dir = np.array([-long_dir[1], long_dir[0]], dtype=np.float64)
    all_long = points @ long_dir
    all_short = points @ short_dir
    long_keep, _, _ = _densest_fixed_window(all_long, long_m + 2.0 * config.containment_tolerance_m)
    short_keep, _, _ = _densest_fixed_window(all_short, short_m + 2.0 * config.containment_tolerance_m)
    keep = long_keep & short_keep
    retained_points = int(np.count_nonzero(keep))
    if retained_points < 3:
        empty_side = _Side(0.0, 0, 0.0, 0.0, False, False)
        empty_axis = _Axis("underconstrained", 0.0, 0.0, 0.0, empty_side, empty_side, 0.0)
        return _Candidate(theta, math.inf, 0.0, 0, empty_axis, empty_axis, 0.0)

    long_values = all_long[keep]
    short_values = all_short[keep]
    border = None if near_border is None else near_border[keep]
    long_axis = _axis_observation(
        long_values,
        short_values,
        long_m,
        short_m,
        border,
        config=config,
    )
    short_axis = _axis_observation(
        short_values,
        long_values,
        short_m,
        long_m,
        border,
        config=config,
    )
    side_ratios = (
        long_axis.low.span_ratio,
        long_axis.high.span_ratio,
        short_axis.low.span_ratio,
        short_axis.high.span_ratio,
    )
    edge_quality = float(np.mean(np.clip(side_ratios, 0.0, 1.0)))
    retained_fraction = float(retained_points / points.shape[0])
    long_oversize = max(0.0, long_axis.observed_span_m - long_m)
    short_oversize = max(0.0, short_axis.observed_span_m - short_m)
    scale = max(config.containment_tolerance_m, 1e-6)
    oversize_loss = (long_oversize / scale) ** 2 + (short_oversize / scale) ** 2

    # Containment rejects clutter/outliers; long side-spanning termination
    # evidence disambiguates orientation without treating the image border as
    # a box edge.  Missing/cropped sides lower confidence but do not change
    # the fixed dimensions.
    score = (
        0.80 * (1.0 - retained_fraction)
        + 0.32 * (1.0 - edge_quality)
        + 0.20 * oversize_loss
    )
    return _Candidate(
        theta=theta,
        score=float(score),
        retained_fraction=retained_fraction,
        retained_points=retained_points,
        long_axis=long_axis,
        short_axis=short_axis,
        edge_quality=edge_quality,
    )


def _search(
    points: FloatArray,
    near_border: NDArray[np.bool_] | None,
    long_m: float,
    short_m: float,
    config: RectangleFitConfig,
) -> _Candidate:
    coarse_step = math.radians(config.coarse_angle_step_deg)
    coarse_angles = np.arange(0.0, math.pi, coarse_step, dtype=np.float64)
    best_coarse = min(
        (
            _candidate(points, near_border, float(theta), long_m, short_m, config)
            for theta in coarse_angles
        ),
        key=lambda item: (item.score, item.theta),
    )
    fine_step = math.radians(config.fine_angle_step_deg)
    half_width = math.radians(config.fine_half_width_deg)
    offsets = np.arange(-half_width, half_width + 0.5 * fine_step, fine_step)
    return min(
        (
            _candidate(
                points,
                near_border,
                best_coarse.theta + float(offset),
                long_m,
                short_m,
                config,
            )
            for offset in offsets
        ),
        key=lambda item: (item.score, item.theta),
    )


def _failure(reason: str, *, count: int = 0) -> RectangleFitResult:
    return RectangleFitResult(
        center_xy_m=None,
        yaw_rad=None,
        long_axis_xy=None,
        short_axis_xy=None,
        corners_xy_m=None,
        observability={
            "center_long": "underconstrained",
            "center_short": "underconstrained",
            "yaw": "underconstrained",
            "reference": "underconstrained",
        },
        feasible_intervals=(),
        side_support={},
        score=math.inf,
        second_score=math.inf,
        candidate_margin=0.0,
        yaw_curvature=0.0,
        retained_fraction=0.0,
        reasons=(reason,),
        diagnostics={"input_points": int(count)},
    )


def fit_fixed_rectangle(
    points_xy: ArrayLike,
    box_model: BoxModel | None = None,
    *,
    pixels_uv: ArrayLike | None = None,
    image_shape: tuple[int, int] | None = None,
    config: RectangleFitConfig | None = None,
) -> RectangleFitResult:
    """Fit the known parcel footprint and expose crop observability.

    ``pixels_uv`` must refer to the same raw-depth pixels as ``points_xy``.
    Without pixels a complete metric extent can still prove two physical
    edges, but a shorter extent is conservatively underconstrained because
    the missing side cannot be identified as image-censored.
    """

    model = BoxModel() if box_model is None else box_model
    settings = RectangleFitConfig() if config is None else config
    points = _validate_points(points_xy)
    pixels: FloatArray | None = None
    if pixels_uv is not None:
        raw_pixels = np.asarray(pixels_uv, dtype=np.float64)
        if raw_pixels.shape != np.asarray(points_xy).shape:
            raise ValueError("pixels_uv must have the same (N, 2) shape as points_xy")
        finite = np.all(np.isfinite(np.asarray(points_xy, dtype=np.float64)), axis=1)
        pixels = raw_pixels[finite]
    if points.shape[0] < settings.min_points:
        return _failure("insufficient_rectangle_points", count=points.shape[0])
    points, pixels = _even_sample(points, pixels, settings.max_fit_points)
    near_border = _near_border(pixels, image_shape, settings.border_margin_px)

    long_m = float(model.long_m)
    short_m = float(model.short_m)
    if not (long_m > short_m > 0.0):
        raise ValueError("box model requires long_m > short_m > 0")

    best = _search(points, near_border, long_m, short_m, settings)
    orthogonal = _candidate(
        points,
        near_border,
        best.theta + math.pi / 2.0,
        long_m,
        short_m,
        settings,
    )
    assignment_margin = max(0.0, float(orthogonal.score - best.score))
    delta = math.radians(max(0.5, settings.coarse_angle_step_deg * 0.5))
    left = _candidate(points, near_border, best.theta - delta, long_m, short_m, settings)
    right = _candidate(points, near_border, best.theta + delta, long_m, short_m, settings)
    curvature = max(0.0, float(min(left.score, right.score) - best.score))

    reasons: list[str] = []
    yaw_valid = True
    if assignment_margin < settings.min_axis_assignment_margin:
        reasons.append("axis_90_ambiguous")
        yaw_valid = False
    if curvature < settings.min_yaw_curvature:
        reasons.append("yaw_ill_conditioned")
        yaw_valid = False

    for name, axis in (("center_long", best.long_axis), ("center_short", best.short_axis)):
        if axis.status == "underconstrained":
            reasons.append(f"{name}_underconstrained")

    long_dir = np.array([math.cos(best.theta), math.sin(best.theta)], dtype=np.float64)
    short_dir = np.array([-long_dir[1], long_dir[0]], dtype=np.float64)
    center_candidate = best.long_axis.center_m * long_dir + best.short_axis.center_m * short_dir
    center_valid = (
        yaw_valid
        and best.long_axis.status != "underconstrained"
        and best.short_axis.status != "underconstrained"
    )
    center = tuple(float(value) for value in center_candidate) if center_valid else None
    yaw = _line_normalize(best.theta) if yaw_valid else None
    long_axis = tuple(float(value) for value in long_dir) if yaw_valid else None
    short_axis = tuple(float(value) for value in short_dir) if yaw_valid else None
    corners: tuple[tuple[float, float], ...] | None = None
    if center_valid:
        corner_array = np.stack(
            [
                center_candidate + sx * 0.5 * long_m * long_dir + sy * 0.5 * short_m * short_dir
                for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
            ]
        )
        corners = tuple(tuple(float(value) for value in row) for row in corner_array)

    feasible: list[AxisFeasibleInterval] = []
    if best.long_axis.status == "underconstrained":
        feasible.append(
            AxisFeasibleInterval(
                "center_long",
                best.long_axis.feasible_lower_m,
                best.long_axis.feasible_upper_m,
            )
        )
    if best.short_axis.status == "underconstrained":
        feasible.append(
            AxisFeasibleInterval(
                "center_short",
                best.short_axis.feasible_lower_m,
                best.short_axis.feasible_upper_m,
            )
        )

    support = {
        "long_low": best.long_axis.low.to_dict(),
        "long_high": best.long_axis.high.to_dict(),
        "short_low": best.short_axis.low.to_dict(),
        "short_high": best.short_axis.high.to_dict(),
    }
    return RectangleFitResult(
        center_xy_m=center,
        yaw_rad=yaw,
        long_axis_xy=long_axis,
        short_axis_xy=short_axis,
        corners_xy_m=corners,
        observability={
            "center_long": best.long_axis.status,
            "center_short": best.short_axis.status,
            "yaw": "constrained" if yaw_valid else "underconstrained",
            "reference": "constrained" if yaw_valid else "underconstrained",
        },
        feasible_intervals=tuple(feasible),
        side_support=support,
        score=float(best.score),
        second_score=float(orthogonal.score),
        candidate_margin=assignment_margin,
        yaw_curvature=curvature,
        retained_fraction=float(best.retained_fraction),
        reasons=tuple(dict.fromkeys(reasons)),
        diagnostics={
            "input_points": int(points.shape[0]),
            "retained_points": best.retained_points,
            "edge_quality": best.edge_quality,
            "long_observed_span_m": best.long_axis.observed_span_m,
            "short_observed_span_m": best.short_axis.observed_span_m,
            "long_center_feasible_m": [best.long_axis.feasible_lower_m, best.long_axis.feasible_upper_m],
            "short_center_feasible_m": [best.short_axis.feasible_lower_m, best.short_axis.feasible_upper_m],
            "best_angle_deg": math.degrees(best.theta),
            "orthogonal_angle_deg": math.degrees(_line_normalize(best.theta + math.pi / 2.0)),
            "side_support": support,
            "best_second_margin": assignment_margin,
            "condition": {"yaw_curvature": curvature},
            "crop_state": {
                "center_long": best.long_axis.status,
                "center_short": best.short_axis.status,
            },
        },
    )

