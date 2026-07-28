from __future__ import annotations

import math
import sys

import numpy as np
import pytest

from parcel_pose.estimator import ParcelPoseEstimator
from parcel_pose.models import (
    BoxDimensionPrior,
    BoxModel,
    Calibration,
    CalibrationState,
    CameraIntrinsics,
    EstimatorConfig,
    Plane,
)
from parcel_pose.output import pose_estimate_to_dict
from parcel_pose.projection import project_points_to_plane, unproject_plane_points
from parcel_pose.visualization import project_points_to_pixels

from .synthetic_scene import DEPTH_SCALE_M, line_angle_error_deg, tilted_scene


MEASURED_BOXES_M = (
    (0.400, 0.253, 0.160),
    (0.395, 0.252, 0.164),
    (0.395, 0.254, 0.164),
    (0.400, 0.256, 0.161),
    (0.401, 0.252, 0.159),
    (0.401, 0.255, 0.156),
    (0.399, 0.253, 0.160),
    (0.400, 0.253, 0.159),
)


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


def test_estimator_reuses_fixed_top_plane_geometry_across_frames() -> None:
    scene = tilted_scene(yaw_deg=23.0, seed=132)
    estimator = _partial_estimator(scene)
    projector = estimator._top_projector

    first = estimator.estimate(scene.depth_z16, depth_scale=DEPTH_SCALE_M)
    second = estimator.estimate(scene.depth_z16, depth_scale=DEPTH_SCALE_M)

    assert estimator._top_projector is projector
    assert pose_estimate_to_dict(first) == pose_estimate_to_dict(second)


@pytest.mark.parametrize(
    ("intrinsics_factory", "plane_frame", "expected_reason"),
    [
        (lambda intr: intr, "unknown_frame", "table_plane_frame_unresolved"),
        (
            lambda intr: CameraIntrinsics(
                width=intr.width,
                height=intr.height,
                fx=intr.fx,
                fy=intr.fy,
                cx=intr.cx,
                cy=intr.cy,
                coeffs=(0.01, 0.0, 0.0, 0.0, 0.0),
            ),
            "depth",
            "invalid_depth_or_metadata",
        ),
    ],
)
def test_cached_geometry_failure_preserves_per_frame_metadata_and_clears_evidence(
    intrinsics_factory,
    plane_frame: str,
    expected_reason: str,
) -> None:
    scene = tilted_scene(yaw_deg=9.0, seed=811)
    intrinsics = intrinsics_factory(_intrinsics(scene))
    calibration = Calibration(
        state=CalibrationState.PLANE_CALIBRATED_PARTIAL,
        table_plane=Plane(
            normal=scene.table_normal,
            d=scene.table_d,
            frame=plane_frame,
        ),
    )
    estimator = ParcelPoseEstimator(intrinsics, calibration)

    first = estimator.estimate(
        scene.depth_z16,
        depth_scale=DEPTH_SCALE_M,
        timestamp_ms=10.0,
        frame_id=1,
    )
    estimator.last_evidence = object()  # type: ignore[assignment]
    second = estimator.estimate(
        scene.depth_z16,
        depth_scale=DEPTH_SCALE_M,
        timestamp_ms=20.0,
        frame_id=2,
    )

    assert first.reasons == (expected_reason,)
    assert second.reasons == (expected_reason,)
    assert (first.timestamp_ms, first.frame_id) == (10.0, 1)
    assert (second.timestamp_ms, second.frame_id) == (20.0, 2)
    assert estimator.last_evidence is None


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


def test_dimension_diagnostics_distinguish_configured_model_from_measured_prior() -> None:
    scene = tilted_scene(yaw_deg=19.0, seed=571)
    configured = _partial_estimator(scene).estimate(
        scene.depth_z16,
        depth_scale=DEPTH_SCALE_M,
    )
    prior = BoxDimensionPrior(
        samples_m=MEASURED_BOXES_M,
        source="manual_tape_measurements_2026-07-28",
    )
    measured = _partial_estimator(
        scene,
        EstimatorConfig(box_dimension_prior=prior),
    ).estimate(
        scene.depth_z16,
        depth_scale=DEPTH_SCALE_M,
    )

    assert (
        configured.diagnostics["box_dimensions"]["inference"]
        == "fixed_configured_model"
    )
    assert (
        measured.diagnostics["box_dimensions"]["inference"]
        == "fixed_population_representative"
    )


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


@pytest.mark.parametrize(
    ("dimensions_m", "yaw_deg"),
    zip(MEASURED_BOXES_M, (-71.0, -46.0, -18.0, 7.0, 29.0, 48.0, 69.0, 86.0), strict=True),
)
def test_population_median_model_covers_each_measured_box(
    dimensions_m: tuple[float, float, float],
    yaw_deg: float,
) -> None:
    long_m, short_m, height_m = dimensions_m
    scene = tilted_scene(
        yaw_deg=yaw_deg,
        center_plane_xy_m=(0.025, -0.018),
        box_long_m=long_m,
        box_short_m=short_m,
        box_height_m=height_m,
        depth_noise_std_m=0.0008,
        hole_rate=0.05,
        seed=900 + int(yaw_deg),
    )

    estimate = _partial_estimator(scene).estimate(
        scene.depth_z16,
        depth_scale=DEPTH_SCALE_M,
    )

    truth_center = project_points_to_plane(
        [scene.center_depth_m],
        _top_plane(scene),
    )[0]
    assert estimate.full_pose_valid, estimate.reasons
    assert np.linalg.norm(np.asarray(estimate.center_plane_xy_m) - truth_center) <= 0.010
    assert line_angle_error_deg(estimate.yaw_rad, scene.yaw_rad) <= 2.0
    assert estimate.box_model.long_m == pytest.approx(0.400)
    assert estimate.box_model.short_m == pytest.approx(0.253)
    assert estimate.box_model.height_m == pytest.approx(0.160)


@pytest.mark.parametrize(
    ("dimensions_m", "yaw_deg"),
    [
        ((0.395, 0.252, 0.164), 31.0),
        ((0.401, 0.255, 0.156), -37.0),
    ],
)
def test_measured_dimension_extremes_keep_overlay_iou_bounded(
    dimensions_m: tuple[float, float, float],
    yaw_deg: float,
) -> None:
    """Lock overlay agreement for noisy, tilted measured-size extremes."""

    cv2 = pytest.importorskip("cv2")
    long_m, short_m, height_m = dimensions_m
    scene = tilted_scene(
        yaw_deg=yaw_deg,
        center_plane_xy_m=(0.025, -0.018),
        box_long_m=long_m,
        box_short_m=short_m,
        box_height_m=height_m,
        depth_noise_std_m=0.0005,
        hole_rate=0.03,
        seed=1_500 + int(height_m * 1_000),
    )
    prior = BoxDimensionPrior(
        samples_m=MEASURED_BOXES_M,
        source="manual_tape_measurements_2026-07-28",
    )
    estimator = _partial_estimator(
        scene,
        EstimatorConfig(box_dimension_prior=prior),
    )

    estimate = estimator.estimate(scene.depth_z16, depth_scale=DEPTH_SCALE_M)

    assert estimate.full_pose_valid, estimate.reasons
    evidence = estimator.last_evidence
    assert evidence is not None
    assert evidence.rectangle.corners_xy_m is not None

    fitted_corners_depth = unproject_plane_points(
        evidence.rectangle.corners_xy_m,
        evidence.projection.plane,
        origin=evidence.projection.origin_3d_m,
        basis=(
            evidence.projection.basis_u_3d,
            evidence.projection.basis_v_3d,
        ),
    )
    fitted_pixels = project_points_to_pixels(fitted_corners_depth, _intrinsics(scene))

    long_axis = (
        math.cos(scene.yaw_rad) * scene.plane_u
        + math.sin(scene.yaw_rad) * scene.plane_v
    )
    short_axis = (
        -math.sin(scene.yaw_rad) * scene.plane_u
        + math.cos(scene.yaw_rad) * scene.plane_v
    )
    truth_corners_depth = np.asarray(
        [
            scene.center_depth_m
            + long_sign * 0.5 * long_m * long_axis
            + short_sign * 0.5 * short_m * short_axis
            for long_sign, short_sign in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ]
    )
    truth_pixels = project_points_to_pixels(truth_corners_depth, _intrinsics(scene))

    fitted_mask = np.zeros(scene.top_mask.shape, dtype=np.uint8)
    truth_mask = np.zeros_like(fitted_mask)
    cv2.fillPoly(fitted_mask, [np.rint(fitted_pixels).astype(np.int32)], 1)
    cv2.fillPoly(truth_mask, [np.rint(truth_pixels).astype(np.int32)], 1)
    intersection = int(np.count_nonzero(fitted_mask & truth_mask))
    union = int(np.count_nonzero(fitted_mask | truth_mask))
    assert intersection / union >= 0.90


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


def test_failed_frame_clears_previous_rectangle_evidence() -> None:
    scene = tilted_scene(yaw_deg=18.0, depth_noise_std_m=0.0005, seed=190)
    estimator = _partial_estimator(scene)

    first = estimator.estimate(scene.depth_z16, depth_scale=DEPTH_SCALE_M)
    assert first.full_pose_valid
    assert estimator.last_evidence is not None

    second = estimator.estimate(
        np.zeros_like(scene.depth_z16),
        depth_scale=DEPTH_SCALE_M,
    )
    assert not second.full_pose_valid
    assert estimator.last_evidence is None


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
        scene.center_depth_m - 0.5 * scene.box_height_m * scene.table_normal,
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
        scene.center_depth_m - 0.5 * scene.box_height_m * scene.table_normal,
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


def _polygon_area_px(points_uv: np.ndarray) -> float:
    points = np.asarray(points_uv, dtype=np.float64)
    return 0.5 * abs(
        float(
            np.dot(points[:, 0], np.roll(points[:, 1], -1))
            - np.dot(points[:, 1], np.roll(points[:, 0], -1))
        )
    )


def _best_cyclic_corner_errors_px(
    actual_uv: np.ndarray,
    expected_uv: np.ndarray,
) -> np.ndarray:
    actual = np.asarray(actual_uv, dtype=np.float64)
    expected = np.asarray(expected_uv, dtype=np.float64)
    candidates: list[np.ndarray] = []
    for ordered in (expected, expected[::-1]):
        for shift in range(4):
            delta = actual - np.roll(ordered, shift, axis=0)
            candidates.append(np.linalg.norm(delta, axis=1))
    return min(candidates, key=lambda errors: float(np.mean(np.square(errors))))


@pytest.mark.parametrize("actual_height_m", [0.156, 0.164])
def test_fixed_model_height_extremes_have_predictable_overlay_and_center_z_bias(
    actual_height_m: float,
) -> None:
    """Verify the explicit limitation of one population height at both extremes.

    This uses exact synthetic projective geometry; it does not claim that the
    nominal robot camera chain is independently validated real-world ground
    truth.  The estimator deliberately keeps a 400 x 253 x 160 mm model rather
    than resizing an individual box on every frame.
    """

    model = BoxModel(long_m=0.400, short_m=0.253, height_m=0.160)
    scene = tilted_scene(
        yaw_deg=27.0,
        center_plane_xy_m=(0.035, -0.025),
        normal=(0.0, 0.0, -1.0),
        box_long_m=model.long_m,
        box_short_m=model.short_m,
        box_height_m=actual_height_m,
        depth_noise_std_m=0.0,
        hole_rate=0.0,
        seed=77,
    )

    # A proper 180-degree rotation maps the synthetic table normal to base +Z.
    transform_base_from_depth = np.eye(4, dtype=np.float64)
    transform_base_from_depth[:3, :3] = np.diag((1.0, -1.0, -1.0))
    transform_base_from_depth[:3, 3] = (0.30, -0.10, 1.50)
    calibration = Calibration(
        state=CalibrationState.BASE_VALIDATED,
        table_plane=_table_plane(scene),
        T_base_from_head=np.eye(4),
        T_head_from_depth=transform_base_from_depth,
    )
    estimator = ParcelPoseEstimator(
        _intrinsics(scene),
        calibration,
        EstimatorConfig(box_model=model),
    )
    estimate = estimator.estimate(scene.depth_z16, depth_scale=DEPTH_SCALE_M)

    assert estimate.full_pose_valid, estimate.reasons
    assert estimate.absolute_valid
    assert estimate.center_depth_m is not None
    assert estimate.box_center_base_xyz_m is not None
    evidence = estimator.last_evidence
    assert evidence is not None
    assert evidence.rectangle.corners_xy_m is not None

    overlay_corners_depth = unproject_plane_points(
        evidence.rectangle.corners_xy_m,
        evidence.projection.plane,
        origin=evidence.projection.origin_3d_m,
        basis=(
            evidence.projection.basis_u_3d,
            evidence.projection.basis_v_3d,
        ),
    )
    overlay_corners_uv = project_points_to_pixels(
        overlay_corners_depth,
        _intrinsics(scene),
    )

    long_axis = (
        math.cos(scene.yaw_rad) * scene.plane_u
        + math.sin(scene.yaw_rad) * scene.plane_v
    )
    short_axis = (
        -math.sin(scene.yaw_rad) * scene.plane_u
        + math.cos(scene.yaw_rad) * scene.plane_v
    )
    signs = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
    true_corners_depth = np.asarray(
        [
            scene.center_depth_m
            + long_sign * 0.5 * scene.box_long_m * long_axis
            + short_sign * 0.5 * scene.box_short_m * short_axis
            for long_sign, short_sign in signs
        ]
    )
    true_corners_uv = project_points_to_pixels(
        true_corners_depth,
        _intrinsics(scene),
    )

    # For parallel fronto-planar surfaces, the fixed overlay's pixel-area
    # ratio is exactly the square of the true/model camera-depth ratio.
    overlay_area_ratio = _polygon_area_px(overlay_corners_uv) / _polygon_area_px(
        true_corners_uv
    )
    expected_area_ratio = (
        float(scene.center_depth_m[2]) / float(estimate.center_depth_m[2])
    ) ** 2
    assert overlay_area_ratio == pytest.approx(expected_area_ratio, abs=1e-10)
    assert (overlay_area_ratio > 1.0) is (actual_height_m < model.height_m)
    corner_errors_px = _best_cyclic_corner_errors_px(
        overlay_corners_uv,
        true_corners_uv,
    )
    assert float(np.max(corner_errors_px)) <= 3.0

    table_point_depth = scene.table_normal * scene.table_d
    table_point_base = (
        transform_base_from_depth[:3, :3] @ table_point_depth
        + transform_base_from_depth[:3, 3]
    )
    expected_model_center_z = float(table_point_base[2]) + 0.5 * model.height_m
    true_center_depth = (
        scene.center_depth_m - 0.5 * actual_height_m * scene.table_normal
    )
    true_center_base = (
        transform_base_from_depth[:3, :3] @ true_center_depth
        + transform_base_from_depth[:3, 3]
    )
    estimated_center_z = float(estimate.box_center_base_xyz_m[2])

    assert estimated_center_z == pytest.approx(expected_model_center_z, abs=1e-12)
    assert estimated_center_z - float(true_center_base[2]) == pytest.approx(
        0.5 * (model.height_m - actual_height_m),
        abs=1e-12,
    )
