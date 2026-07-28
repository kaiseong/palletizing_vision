from __future__ import annotations

import math

import numpy as np
import pytest

from parcel_pose.rectangle_fit import (
    _densest_fixed_window,
    _linear_quantile_pair,
    fit_fixed_rectangle,
)

from .synthetic_scene import (
    line_angle_error_deg,
    orthographic_image_observation,
    rectangle_support_points,
)


def _reference_densest_fixed_window(
    values: np.ndarray,
    width: float,
) -> tuple[np.ndarray, float, float]:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    right = np.searchsorted(sorted_values, sorted_values + width, side="right")
    counts = right - np.arange(sorted_values.size)
    start = int(np.argmax(counts))
    stop = int(right[start])
    low = float(sorted_values[start])
    high = low + width
    mask = (values >= low) & (values <= high)
    if int(np.count_nonzero(mask)) < stop - start:
        mask[order[start:stop]] = True
    return mask, low, high


@pytest.mark.parametrize("seed", range(12))
def test_direct_value_sort_matches_reference_window(seed: int) -> None:
    rng = np.random.default_rng(seed)
    values = np.round(rng.normal(0.0, 0.18, 6_000), decimals=4)
    width = float(rng.uniform(0.20, 0.45))

    actual = _densest_fixed_window(values, width)
    expected = _reference_densest_fixed_window(values, width)

    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1:] == expected[1:]


def test_direct_value_sort_preserves_boundary_ties() -> None:
    values = np.array([-0.2, -0.2, 0.0, 0.2, 0.2, 0.4], dtype=np.float64)

    actual = _densest_fixed_window(values, 0.4)
    expected = _reference_densest_fixed_window(values, 0.4)

    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1:] == expected[1:]


def test_direct_value_sort_floating_guard_reconstructs_source_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = np.array([-0.2, -0.1, 0.0, 0.1, 0.2], dtype=np.float64)
    original_count_nonzero = np.count_nonzero

    def undercount_once(value: object) -> int:
        return max(0, int(original_count_nonzero(value)) - 1)

    monkeypatch.setattr("parcel_pose.rectangle_fit.np.count_nonzero", undercount_once)
    mask, low, high = _densest_fixed_window(values, 0.2)

    assert int(original_count_nonzero(mask)) >= 3
    assert low == -0.2
    assert high == 0.0


@pytest.mark.parametrize("size", [1, 2, 4, 19, 20, 21, 100, 6_000])
@pytest.mark.parametrize("quantiles", [(0.004, 0.996), (0.03, 0.97), (0.25, 0.75)])
def test_linear_quantile_pair_matches_numpy_linear_method(
    size: int,
    quantiles: tuple[float, float],
) -> None:
    rng = np.random.default_rng(size)
    values = np.round(rng.normal(0.0, 0.4, size), decimals=4)

    actual = _linear_quantile_pair(values, *quantiles)
    expected = np.quantile(values, quantiles, method="linear")

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("yaw_deg", [-80.0, -47.0, -5.0, 0.0, 23.0, 44.0, 68.0, 89.0])
def test_full_fixed_rectangle_meets_clean_numeric_targets(yaw_deg: float) -> None:
    truth_center = np.array([0.035, -0.042], dtype=np.float64)
    points = rectangle_support_points(
        center_xy_m=tuple(truth_center),
        yaw_deg=yaw_deg,
        points_per_edge=220,
        interior_points=1_800,
        noise_std_m=0.0006,
        seed=int(yaw_deg) + 100,
    )

    result = fit_fixed_rectangle(points)

    assert result.full_pose_valid, result.reasons
    assert result.center_xy_m is not None
    assert result.yaw_rad is not None
    assert np.linalg.norm(np.asarray(result.center_xy_m) - truth_center) <= 0.010
    assert line_angle_error_deg(result.yaw_rad, math.radians(yaw_deg)) <= 1.5
    assert result.observability["center_long"] == "both_edges"
    assert result.observability["center_short"] == "both_edges"
    assert result.observability["yaw"] == "constrained"
    assert result.candidate_margin > 0.0
    assert result.yaw_curvature > 0.0
    assert result.diagnostics["long_observed_span_m"] == pytest.approx(0.400, abs=0.025)
    assert result.diagnostics["short_observed_span_m"] == pytest.approx(0.253, abs=0.025)


def test_fixed_fit_tolerates_depth_like_holes_and_metric_outliers() -> None:
    truth_center = np.array([0.040, -0.030])
    truth_yaw = math.radians(37.0)
    points = rectangle_support_points(
        center_xy_m=tuple(truth_center),
        yaw_deg=37.0,
        points_per_edge=350,
        interior_points=3_000,
        noise_std_m=0.0015,
        hole_rate=0.35,
        outlier_count=180,
        seed=33,
    )

    result = fit_fixed_rectangle(points)

    assert result.full_pose_valid, result.reasons
    assert np.linalg.norm(np.asarray(result.center_xy_m) - truth_center) <= 0.010
    assert line_angle_error_deg(result.yaw_rad, truth_yaw) <= 1.5
    assert result.retained_fraction < 1.0
    assert result.retained_fraction > 0.85


@pytest.mark.parametrize(
    ("truth_center", "expected_axis", "expected_censored_side"),
    [
        ((0.180, 0.0), "center_long", "long_high"),
        ((-0.180, 0.0), "center_long", "long_low"),
        ((0.0, 0.140), "center_short", "short_high"),
        ((0.0, -0.140), "center_short", "short_low"),
    ],
)
def test_each_one_edge_crop_is_inferred_only_from_known_border_side(
    truth_center: tuple[float, float],
    expected_axis: str,
    expected_censored_side: str,
) -> None:
    points = rectangle_support_points(
        center_xy_m=truth_center,
        yaw_deg=0.0,
        points_per_edge=500,
        interior_points=6_000,
        noise_std_m=0.0003,
        seed=53,
    )
    observed, pixels_uv, image_shape = orthographic_image_observation(points)

    result = fit_fixed_rectangle(observed, pixels_uv=pixels_uv, image_shape=image_shape)

    assert result.full_pose_valid, result.reasons
    assert result.observability[expected_axis] == "one_edge_inferred"
    other_axis = "center_short" if expected_axis == "center_long" else "center_long"
    assert result.observability[other_axis] == "both_edges"
    assert result.side_support[expected_censored_side]["censored"] is True
    assert result.center_xy_m is not None
    assert np.linalg.norm(np.asarray(result.center_xy_m) - np.asarray(truth_center)) <= 0.020
    assert line_angle_error_deg(result.yaw_rad, 0.0) <= 1.5


@pytest.mark.parametrize(
    ("pixels_per_meter", "image_shape", "principal_uv", "underconstrained_axis"),
    [
        (2_000.0, (600, 640), (320.0, 300.0), "center_long"),
        (1_000.0, (180, 640), (320.0, 90.0), "center_short"),
    ],
)
def test_opposite_edges_missing_returns_null_center_and_feasible_interval(
    pixels_per_meter: float,
    image_shape: tuple[int, int],
    principal_uv: tuple[float, float],
    underconstrained_axis: str,
) -> None:
    points = rectangle_support_points(
        yaw_deg=0.0,
        points_per_edge=900,
        interior_points=14_000,
        noise_std_m=0.00025,
        seed=71,
    )
    observed, pixels_uv, image_shape = orthographic_image_observation(
        points,
        pixels_per_meter=pixels_per_meter,
        image_shape=image_shape,
        principal_uv=principal_uv,
    )

    result = fit_fixed_rectangle(observed, pixels_uv=pixels_uv, image_shape=image_shape)

    assert result.center_xy_m is None
    assert result.yaw_rad is not None
    assert result.full_pose_valid is False
    assert result.observability[underconstrained_axis] == "underconstrained"
    assert f"{underconstrained_axis}_underconstrained" in result.reasons
    intervals = {item.axis: item for item in result.feasible_intervals}
    assert underconstrained_axis in intervals
    interval = intervals[underconstrained_axis]
    assert interval.lower_m < 0.0 < interval.upper_m
    assert interval.upper_m - interval.lower_m >= 0.030


def test_single_unidentified_edge_abstains_on_90_degree_axis_swap() -> None:
    rng = np.random.default_rng(83)
    theta = math.radians(17.0)
    direction = np.array([math.cos(theta), math.sin(theta)])
    normal = np.array([-direction[1], direction[0]])
    parameter = np.linspace(-0.075, 0.075, 900)
    points = parameter[:, None] * direction
    points += rng.normal(0.0, 0.0003, (points.shape[0], 1)) * normal

    result = fit_fixed_rectangle(points)

    assert result.yaw_rad is None
    assert result.center_xy_m is None
    assert result.full_pose_valid is False
    assert "axis_90_ambiguous" in result.reasons
    assert result.observability["yaw"] == "underconstrained"
    assert result.observability["reference"] == "underconstrained"
    assert result.candidate_margin < 0.045


def test_corner_only_support_never_manufactures_a_full_pose() -> None:
    rng = np.random.default_rng(97)
    horizontal = np.column_stack((np.linspace(0.0, 0.055, 300), np.zeros(300)))
    vertical = np.column_stack((np.zeros(300), np.linspace(0.0, 0.055, 300)))
    points = np.concatenate((horizontal, vertical), axis=0)
    points += rng.normal(0.0, 0.00035, points.shape)

    result = fit_fixed_rectangle(points)

    assert result.full_pose_valid is False
    assert result.center_xy_m is None or result.yaw_rad is None
    assert result.reasons
