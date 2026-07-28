from __future__ import annotations

import math

import numpy as np
import pytest

from parcel_pose.models import CameraIntrinsics, Plane
from parcel_pose.plane import (
    fit_plane_ransac,
    fit_plane_svd,
    offset_plane,
    orient_plane_toward,
    point_on_plane,
    signed_distances,
)
from parcel_pose.projection import (
    DepthPlaneProjector,
    deproject_depth,
    depth_to_meters,
    intersect_rays_with_plane,
    pixel_rays,
    project_depth_to_plane,
)

from .synthetic_scene import DEPTH_SCALE_M, noisy_plane_points, tilted_scene


def _normal_error_deg(actual: np.ndarray, expected: np.ndarray) -> float:
    cosine = float(np.clip(np.asarray(actual) @ np.asarray(expected), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _intrinsics(scene) -> CameraIntrinsics:
    intr = scene.intrinsics
    return CameraIntrinsics(
        width=intr.width,
        height=intr.height,
        fx=intr.fx,
        fy=intr.fy,
        cx=intr.cx,
        cy=intr.cy,
        fps=30,
    )


def test_svd_recovers_clean_tilted_plane() -> None:
    points, truth_normal, truth_d, _ = noisy_plane_points(
        outlier_count=0,
        noise_std_m=0.0004,
        seed=3,
    )

    fitted = fit_plane_svd(points, camera_origin=np.zeros(3))

    assert _normal_error_deg(fitted.normal, truth_normal) < 0.15
    assert abs(fitted.d - truth_d) < 0.001
    assert float(np.sqrt(np.mean(np.square(signed_distances(points, fitted))))) < 0.001


def test_ransac_recovers_plane_and_rejects_injected_outliers() -> None:
    points, truth_normal, truth_d, true_inliers = noisy_plane_points(
        inlier_count=900,
        outlier_count=180,
        noise_std_m=0.0007,
        seed=23,
    )

    result = fit_plane_ransac(
        points,
        tolerance_m=0.003,
        iterations=400,
        min_inliers=700,
        seed=41,
    )

    assert _normal_error_deg(result.plane.normal, truth_normal) < 0.20
    assert abs(result.plane.d - truth_d) < 0.0015
    assert result.inlier_rms_m < 0.0012
    assert result.inlier_ratio > 0.80
    assert np.count_nonzero(result.inlier_mask & true_inliers) > 850
    assert np.count_nonzero(result.inlier_mask & ~true_inliers) < 10


def test_normal_sign_and_150mm_offset_follow_plane_normal_not_frame_z() -> None:
    scene = tilted_scene(depth_noise_std_m=0.0)
    point_table = scene.table_normal * scene.table_d
    camera_origin = np.zeros(3)

    positive_input = Plane(normal=scene.table_normal, d=scene.table_d, frame="depth")
    negative_input = Plane(normal=-scene.table_normal, d=-scene.table_d, frame="depth")
    oriented_a = orient_plane_toward(positive_input, camera_origin, point=point_table)
    oriented_b = orient_plane_toward(negative_input, camera_origin, point=point_table)

    for oriented in (oriented_a, oriented_b):
        assert float(oriented.normal @ (camera_origin - point_table)) > 0.0
        np.testing.assert_allclose(oriented.normal, scene.table_normal, atol=1e-12)
        assert oriented.d == pytest.approx(scene.table_d, abs=1e-12)

        top = offset_plane(oriented, 0.150)
        assert top.d == pytest.approx(oriented.d + 0.150, abs=1e-12)
        np.testing.assert_allclose(
            point_on_plane(top) - point_on_plane(oriented),
            0.150 * oriented.normal,
            atol=1e-12,
        )
        # A tilted normal means a physical normal offset is not a Z-only edit.
        assert abs(float((point_on_plane(top) - point_on_plane(oriented))[0])) > 0.005


def test_ray_plane_intersection_rejects_parallel_and_behind_camera() -> None:
    plane = Plane(normal=np.array([0.0, 0.0, -1.0]), d=-0.75, frame="depth")
    rays = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.2, -0.1, 1.0],
            [1.0, 0.0, 0.0],  # parallel
            [0.0, 0.0, -1.0],  # intersection behind camera
        ],
        dtype=np.float64,
    )

    points, valid = intersect_rays_with_plane(rays, plane)

    np.testing.assert_array_equal(valid, [True, True, False, False])
    np.testing.assert_allclose(points[0], [0.0, 0.0, 0.75], atol=1e-12)
    np.testing.assert_allclose(points[1], [0.15, -0.075, 0.75], atol=1e-12)
    assert np.all(np.isnan(points[~valid]))


def test_raw_z16_conversion_and_deprojection_use_supplied_depth_intrinsics() -> None:
    intrinsics = CameraIntrinsics(width=3, height=2, fx=2.0, fy=4.0, cx=1.0, cy=0.5, fps=30)
    raw = np.array([[1000, 2000, 0], [500, 1500, 2500]], dtype=np.uint16)

    meters = depth_to_meters(raw, 0.001)
    points = deproject_depth(raw, intrinsics, depth_scale=0.001)

    np.testing.assert_allclose(meters, [[1.0, 2.0, 0.0], [0.5, 1.5, 2.5]])
    np.testing.assert_allclose(points[0, 0], [-0.5, -0.125, 1.0])
    np.testing.assert_allclose(points[1, 2], [1.25, 0.3125, 2.5])
    assert np.all(np.isnan(points[0, 2]))
    rays = pixel_rays([[1.0, 0.5], [2.0, 1.5]], intrinsics)
    np.testing.assert_allclose(rays, [[0.0, 0.0, 1.0], [0.5, 0.25, 1.0]])


def test_raw_depth_deprojection_rejects_unhandled_nonzero_distortion() -> None:
    intrinsics = CameraIntrinsics(
        width=2,
        height=2,
        fx=100.0,
        fy=100.0,
        cx=0.5,
        cy=0.5,
        distortion_model="inverse_brown_conrady",
        coeffs=(0.01, 0.0, 0.0, 0.0, 0.0),
    )

    with pytest.raises(ValueError, match="non-zero distortion"):
        deproject_depth(np.ones((2, 2), dtype=np.float64), intrinsics)


def test_fixed_plane_projector_preserves_row_major_sampling_and_metric_geometry() -> None:
    intrinsics = CameraIntrinsics(
        width=4,
        height=3,
        fx=2.0,
        fy=2.0,
        cx=1.5,
        cy=1.0,
    )
    plane = Plane(normal=np.array([0.1, -0.2, -1.0]), d=-1.0, frame="depth")
    depth = np.array(
        [
            [1.00, 0.98, 0.00, 1.04],
            [1.03, 0.99, 1.01, 2.50],
            [np.nan, 0.97, 1.02, 1.00],
        ],
        dtype=np.float64,
    )
    support = np.array(
        [
            [True, True, True, False],
            [True, True, True, True],
            [True, False, True, True],
        ],
        dtype=np.bool_,
    )

    projector = DepthPlaneProjector(intrinsics, plane)
    projected = projector.project(
        depth,
        slab_tolerance_m=0.20,
        min_depth_m=0.20,
        max_depth_m=2.0,
        support_mask=support,
        max_points=4,
    )

    expected_mask = (
        np.isfinite(depth)
        & (depth >= 0.20)
        & (depth <= 2.0)
        & support
        & (np.abs(signed_distances(deproject_depth(depth, intrinsics), plane)) <= 0.20)
    )
    np.testing.assert_array_equal(projected.mask, expected_mask)
    rows, cols = np.nonzero(expected_mask)
    sampled = np.linspace(0, rows.size - 1, 4, dtype=np.int64)
    expected_pixels = np.column_stack((cols[sampled], rows[sampled])).astype(np.float64)
    np.testing.assert_array_equal(projected.pixels_uv, expected_pixels)
    np.testing.assert_allclose(
        signed_distances(projected.points_3d_m, plane),
        0.0,
        rtol=0.0,
        atol=1e-12,
    )
    assert projected.diagnostics["slab_pixels"] == int(np.count_nonzero(expected_mask))
    assert projected.count == 4


def test_top_plane_slab_survives_holes_and_outliers_without_selecting_table() -> None:
    scene = tilted_scene(
        yaw_deg=31.0,
        depth_noise_std_m=0.0012,
        hole_rate=0.18,
        outlier_rate=0.08,
        seed=91,
    )
    top_plane = Plane(normal=scene.top_normal, d=scene.top_d, frame="depth")

    projected = project_depth_to_plane(
        scene.depth_z16,
        _intrinsics(scene),
        top_plane,
        depth_scale=DEPTH_SCALE_M,
        slab_tolerance_m=0.012,
        max_points=None,
    )

    assert projected.count > 8_000
    assert projected.count < int(scene.top_mask.sum())
    assert np.count_nonzero(projected.mask & ~scene.top_mask) == 0
    assert float(np.max(np.abs(signed_distances(projected.points_3d_m, top_plane)))) < 1e-10
    assert projected.diagnostics["slab_pixels"] == projected.count
