from __future__ import annotations

import math
import sys

import numpy as np
import pytest

from parcel_pose.estimator import ParcelPoseEstimator
from parcel_pose.models import (
    Calibration,
    CalibrationState,
    CameraIntrinsics,
    EstimatorConfig,
    Plane,
)
from parcel_pose.projection import project_points_to_plane

from .synthetic_scene import DEPTH_SCALE_M, line_angle_error_deg, tilted_scene


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


def _table_plane(scene) -> Plane:
    return Plane(normal=scene.table_normal, d=scene.table_d, frame="depth")


def _top_plane(scene) -> Plane:
    return Plane(normal=scene.top_normal, d=scene.top_d, frame="depth")


def _partial_estimator(scene, config: EstimatorConfig | None = None) -> ParcelPoseEstimator:
    calibration = Calibration(
        state=CalibrationState.PLANE_CALIBRATED_PARTIAL,
        table_plane=_table_plane(scene),
    )
    return ParcelPoseEstimator(_intrinsics(scene), calibration, config)


@pytest.mark.parametrize("yaw_deg", [-72.0, -31.0, 0.0, 27.0, 76.0])
def test_full_view_depth_frame_estimator_meets_clean_targets(yaw_deg: float) -> None:
    scene = tilted_scene(
        yaw_deg=yaw_deg,
        center_plane_xy_m=(0.030, -0.020),
        depth_noise_std_m=0.0008,
        hole_rate=0.05,
        outlier_rate=0.02,
        seed=int(yaw_deg) + 140,
    )
    estimator = _partial_estimator(scene)

    estimate = estimator.estimate(
        scene.depth_z16,
        depth_scale=DEPTH_SCALE_M,
        timestamp_ms=12_345.0,
        frame_id=17,
    )

    truth_center_xy = project_points_to_plane(
        [scene.center_depth_m],
        _top_plane(scene),
    )[0]
    assert estimate.geometry_valid, estimate.reasons
    assert estimate.full_pose_valid, estimate.reasons
    assert estimate.center_plane_xy_m is not None
    assert estimate.yaw_rad is not None
    assert np.linalg.norm(np.asarray(estimate.center_plane_xy_m) - truth_center_xy) <= 0.010
    assert line_angle_error_deg(estimate.yaw_rad, scene.yaw_rad) <= 1.5
    assert estimate.timestamp_ms == 12_345.0
    assert estimate.frame_id == 17
    assert estimate.frame == "table_plane"
    assert estimate.base_registration_valid is False
    assert estimate.absolute_valid is False
    assert estimate.center_base_xy_m is None
    assert estimate.top_center_base_xyz_m is None
    assert estimate.box_center_base_xyz_m is None
    assert "absolute_base_transform_unvalidated" in estimate.reasons


def test_noisy_independent_scene_p95_and_coverage_targets() -> None:
    center_errors: list[float] = []
    yaw_errors: list[float] = []
    scene_count = 12
    for index, yaw_deg in enumerate(np.linspace(-82.0, 88.0, scene_count)):
        scene = tilted_scene(
            yaw_deg=float(yaw_deg),
            center_plane_xy_m=(0.040 * math.sin(index), 0.030 * math.cos(index)),
            depth_noise_std_m=0.002,
            hole_rate=0.22,
            outlier_rate=0.07,
            seed=200 + index,
        )
        estimate = _partial_estimator(scene).estimate(
            scene.depth_z16,
            depth_scale=DEPTH_SCALE_M,
        )
        if not estimate.full_pose_valid:
            continue
        truth_center = project_points_to_plane([scene.center_depth_m], _top_plane(scene))[0]
        center_errors.append(float(np.linalg.norm(np.asarray(estimate.center_plane_xy_m) - truth_center)))
        yaw_errors.append(line_angle_error_deg(estimate.yaw_rad, scene.yaw_rad))

    coverage = len(center_errors) / scene_count
    abstention_rate = 1.0 - coverage
    assert coverage >= 0.75, {"coverage": coverage, "abstention_rate": abstention_rate}
    assert np.percentile(center_errors, 95) <= 0.020
    assert np.percentile(yaw_errors, 95) <= 4.0


def test_missing_rgb_support_has_same_depth_only_result() -> None:
    scene = tilted_scene(
        yaw_deg=23.0,
        center_plane_xy_m=(-0.025, 0.020),
        depth_noise_std_m=0.0008,
        hole_rate=0.05,
        seed=443,
    )
    estimator = _partial_estimator(scene)

    depth_only = estimator.estimate(scene.depth_z16, depth_scale=DEPTH_SCALE_M)
    all_rgb_support = estimator.estimate(
        scene.depth_z16,
        depth_scale=DEPTH_SCALE_M,
        rgb_support_mask=np.ones(scene.depth_z16.shape, dtype=np.bool_),
    )

    assert depth_only.full_pose_valid
    assert all_rgb_support.full_pose_valid
    np.testing.assert_allclose(depth_only.center_plane_xy_m, all_rgb_support.center_plane_xy_m, atol=0.0)
    assert depth_only.yaw_rad == pytest.approx(all_rgb_support.yaw_rad, abs=0.0)
    assert depth_only.observability == all_rgb_support.observability


def test_no_valid_depth_fails_cleanly_without_pose_fields() -> None:
    scene = tilted_scene(depth_noise_std_m=0.0)
    estimator = _partial_estimator(scene)
    empty_depth = np.zeros_like(scene.depth_z16)

    estimate = estimator.estimate(empty_depth, depth_scale=DEPTH_SCALE_M)

    assert estimate.geometry_valid is False
    assert estimate.full_pose_valid is False
    assert estimate.center_plane_xy_m is None
    assert estimate.center_depth_m is None
    assert estimate.center_base_xy_m is None
    assert estimate.yaw_rad is None
    assert estimate.yaw_mod_180_deg is None
    assert estimate.reasons == ("insufficient_top_plane_points",)
    assert estimate.diagnostics["valid_depth_pixels"] == 0
    assert estimate.diagnostics["projected_points"] == 0


@pytest.mark.parametrize(
    "calibration",
    [
        lambda plane: Calibration(
            state=CalibrationState.BASE_VALIDATED,
            table_plane=plane,
        ),
        lambda plane: Calibration(
            state=CalibrationState.BASE_VALIDATED,
            table_plane=plane,
            T_base_from_head=np.eye(4),
            T_head_from_color=np.eye(4),
            # E_color_from_depth deliberately missing.
        ),
    ],
)
def test_missing_transform_prerequisite_suppresses_every_base_pose_field(calibration) -> None:
    scene = tilted_scene(yaw_deg=19.0, depth_noise_std_m=0.0006, seed=401)
    estimator = ParcelPoseEstimator(
        _intrinsics(scene),
        calibration(_table_plane(scene)),
    )

    estimate = estimator.estimate(scene.depth_z16, depth_scale=DEPTH_SCALE_M)

    assert estimate.full_pose_valid
    assert estimate.center_plane_xy_m is not None
    assert estimate.yaw_rad is not None
    assert estimate.center_base_xy_m is None
    assert estimate.long_axis_base_xy is None
    assert estimate.short_axis_base_xy is None
    assert estimate.base_registration == "unavailable"
    assert estimate.base_registration_valid is False
    assert estimate.absolute_valid is False
    assert "absolute_base_transform_unvalidated" in estimate.reasons


def test_partial_calibration_with_complete_nominal_chain_keeps_base_output_diagnostic_only() -> None:
    scene = tilted_scene(yaw_deg=-26.0, depth_noise_std_m=0.0006, seed=421)
    calibration = Calibration(
        state=CalibrationState.PLANE_CALIBRATED_PARTIAL,
        table_plane=_table_plane(scene),
        T_base_from_head=np.eye(4),
        T_head_from_depth=np.eye(4),
    )

    estimate = ParcelPoseEstimator(_intrinsics(scene), calibration).estimate(
        scene.depth_z16,
        depth_scale=DEPTH_SCALE_M,
    )

    assert estimate.full_pose_valid
    assert estimate.center_base_xy_m is None
    assert estimate.long_axis_base_xy is None
    assert estimate.base_registration == "nominal_unverified"
    assert estimate.base_registration_valid is False
    assert estimate.absolute_valid is False
    assert "nominal_unverified_base" in estimate.diagnostics
    np.testing.assert_allclose(
        estimate.diagnostics["nominal_unverified_base"]["center_xyz_m"],
        scene.center_depth_m,
        atol=0.010,
    )
    np.testing.assert_allclose(
        estimate.diagnostics["nominal_unverified_base"]["box_center_xyz_m"],
        scene.center_depth_m - 0.5 * 0.150 * scene.table_normal,
        atol=0.010,
    )


def test_base_validated_complete_chain_emits_metric_base_center_and_yaw() -> None:
    scene = tilted_scene(
        yaw_deg=23.0,
        center_plane_xy_m=(0.030, -0.020),
        depth_noise_std_m=0.0007,
        seed=443,
    )
    calibration = Calibration(
        state=CalibrationState.BASE_VALIDATED,
        table_plane=_table_plane(scene),
        T_base_from_head=np.eye(4),
        T_head_from_depth=np.eye(4),
    )

    estimate = ParcelPoseEstimator(_intrinsics(scene), calibration).estimate(
        scene.depth_z16,
        depth_scale=DEPTH_SCALE_M,
    )

    true_long_3d = (
        math.cos(scene.yaw_rad) * scene.plane_u
        + math.sin(scene.yaw_rad) * scene.plane_v
    )
    true_base_yaw = math.atan2(true_long_3d[1], true_long_3d[0]) % math.pi
    assert estimate.full_pose_valid, estimate.reasons
    assert estimate.base_registration_valid
    assert estimate.absolute_valid
    assert estimate.base_registration == "validated"
    assert estimate.frame == "base"
    np.testing.assert_allclose(estimate.center_base_xy_m, scene.center_depth_m[:2], atol=0.010)
    np.testing.assert_allclose(
        estimate.top_center_base_xyz_m,
        scene.center_depth_m,
        atol=0.010,
    )
    np.testing.assert_allclose(
        estimate.box_center_base_xyz_m,
        scene.center_depth_m - 0.5 * 0.150 * scene.table_normal,
        atol=0.010,
    )
    assert line_angle_error_deg(estimate.yaw_rad, true_base_yaw) <= 1.5
    assert estimate.long_axis_base_xy is not None
    assert estimate.short_axis_base_xy is not None
    assert estimate.reasons == ()


def test_crop_fine_search_wrap_does_not_mirror_off_origin_center() -> None:
    scene = tilted_scene(
        yaw_deg=0.0,
        center_plane_xy_m=(0.0, -0.25),
        depth_noise_std_m=0.0008,
        hole_rate=0.05,
        seed=77,
    )

    estimate = _partial_estimator(scene).estimate(
        scene.depth_z16,
        depth_scale=DEPTH_SCALE_M,
    )

    assert estimate.full_pose_valid, estimate.reasons
    assert "one_edge_inferred" in estimate.observability.values()
    assert estimate.center_depth_m is not None
    assert np.linalg.norm(
        np.asarray(estimate.center_depth_m) - scene.center_depth_m
    ) <= 0.020
    assert line_angle_error_deg(estimate.yaw_rad, scene.yaw_rad) <= 1.5


def test_similarly_sized_same_height_components_are_not_full_pose_success() -> None:
    left = tilted_scene(
        yaw_deg=8.0,
        center_plane_xy_m=(0.0, -0.20),
        depth_noise_std_m=0.0004,
        seed=611,
    )
    right = tilted_scene(
        yaw_deg=-11.0,
        center_plane_xy_m=(0.0, 0.20),
        depth_noise_std_m=0.0004,
        seed=612,
    )
    combined_depth = left.depth_z16.copy()
    combined_depth[right.top_mask] = right.depth_z16[right.top_mask]

    estimate = _partial_estimator(left).estimate(
        combined_depth,
        depth_scale=DEPTH_SCALE_M,
    )

    assert "multiple_or_ambiguous_components" in estimate.reasons
    assert estimate.geometry_valid is False
    assert estimate.full_pose_valid is False
    assert estimate.per_field_confidence["yaw"] == 0.0


def test_missing_opencv_component_filter_fails_closed(monkeypatch) -> None:
    scene = tilted_scene(
        yaw_deg=21.0,
        center_plane_xy_m=(0.020, -0.015),
        depth_noise_std_m=0.0005,
        seed=733,
    )
    monkeypatch.setitem(sys.modules, "cv2", None)

    estimate = _partial_estimator(scene).estimate(
        scene.depth_z16,
        depth_scale=DEPTH_SCALE_M,
    )

    assert estimate.reasons == ("component_filter_unavailable",)
    assert estimate.geometry_valid is False
    assert estimate.full_pose_valid is False
    assert estimate.absolute_valid is False
    assert estimate.center_plane_xy_m is None
    assert estimate.center_depth_m is None
    assert estimate.yaw_rad is None
    assert all(value == 0.0 for value in estimate.per_field_confidence.values())
