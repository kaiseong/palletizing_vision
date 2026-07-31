"""Unit coverage for the dependency-neutral picking perception facade."""

from __future__ import annotations

import math

import numpy as np
import pytest

from parcel_pose_common.models import (
    Calibration,
    CalibrationState,
    CameraIntrinsics,
    Plane,
    PoseEstimate,
)
from parcel_pose_picking import box_perception
from parcel_pose_picking.box_perception import (
    perceive_box_pose,
    pose_result_from_base_diagnostic,
    pose_result_from_estimate,
)
from parcel_pose_picking.evaluation import BasePoseDiagnostic


def _base_pose(*, yaw_deg: float = 90.0) -> BasePoseDiagnostic:
    return BasePoseDiagnostic(
        box_center_xyz_m=(0.740, -0.012, 0.200),
        top_center_xyz_m=(0.740, -0.012, 0.320),
        yaw_mod_180_deg=yaw_deg,
        yaw_signed_deg=-90.0,
        canonical_reference_deg=90,
        canonical_residual_deg=0.0,
        registration="validated",
    )


def test_base_diagnostic_adapter_has_explicit_base_units_and_line_yaw() -> None:
    result = pose_result_from_base_diagnostic(
        _base_pose(yaw_deg=270.0),
        timestamp_s=12.25,
        diagnostics={"capture_id": "depth-41"},
    )

    assert result.valid is True
    assert result.reason == ""
    assert result.frame == "base"
    assert result.x_m == 0.740
    assert result.y_m == -0.012
    assert result.yaw_rad == math.pi / 2.0
    assert result.timestamp_s == 12.25
    assert result.diagnostics["capture_id"] == "depth-41"
    assert result.diagnostics["base_pose"]["registration"] == "validated"


def test_missing_base_diagnostic_fails_closed_with_reason_and_provenance() -> None:
    result = pose_result_from_base_diagnostic(
        None,
        timestamp_s=3.0,
        invalid_reason="",
        diagnostics={"capture_id": "depth-42"},
    )

    assert result.valid is False
    assert result.reason == "box_base_pose_unavailable"
    assert (result.x_m, result.y_m, result.yaw_rad) == (None, None, None)
    assert result.timestamp_s == 3.0
    assert result.diagnostics == {"capture_id": "depth-42"}


def test_non_finite_base_diagnostic_becomes_inspectable_invalid_result() -> None:
    result = pose_result_from_base_diagnostic(
        _base_pose(yaw_deg=float("nan")),
        timestamp_s=4.0,
    )

    assert result.valid is False
    assert result.reason == "box_base_pose_invalid"
    assert "non-finite" in result.diagnostics["adapter_error"]


def test_estimate_adapter_preserves_sensor_and_quality_provenance() -> None:
    identity = np.eye(4, dtype=np.float64)
    calibration = Calibration(
        state=CalibrationState.BASE_VALIDATED,
        table_plane=Plane(normal=np.array([0.0, 0.0, 1.0]), d=0.0, frame="base"),
        T_base_from_head=identity,
        T_head_from_depth=identity,
    )
    estimate = PoseEstimate(
        timestamp_ms=1_234.5,
        frame_id=17,
        frame="base",
        center_depth_m=(0.740, -0.012, 0.320),
        yaw_rad=math.pi / 2.0,
        diagnostics={"quality": {"fit_score": 0.98}},
        calibration_state=CalibrationState.BASE_VALIDATED,
        base_registration="validated",
        geometry_valid=True,
        full_pose_valid=True,
        base_registration_valid=True,
        absolute_valid=True,
    )

    result = pose_result_from_estimate(
        estimate,
        calibration,
        timestamp_s=88.0,
        capture_diagnostics={"timestamp_domain": "monotonic"},
    )

    assert result.valid is True
    assert result.frame == "base"
    assert (result.x_m, result.y_m) == (0.740, -0.012)
    assert result.yaw_rad == math.pi / 2.0
    assert result.timestamp_s == 88.0
    assert result.diagnostics["source_timestamp_ms"] == 1_234.5
    assert result.diagnostics["source_frame_id"] == 17
    assert result.diagnostics["source_validity"]["absolute_valid"] is True
    assert result.diagnostics["estimator"]["quality"]["fit_score"] == 0.98
    assert result.diagnostics["capture"]["timestamp_domain"] == "monotonic"


def test_invalid_estimate_uses_stable_source_reason_without_losing_diagnostics() -> None:
    estimate = PoseEstimate(
        timestamp_ms=2_500.0,
        frame_id=9,
        reasons=("insufficient_top_plane_points",),
        diagnostics={"selected_points": 7},
    )

    result = pose_result_from_estimate(
        estimate,
        Calibration(),
        timestamp_s=9.5,
    )

    assert result.valid is False
    assert result.reason == "insufficient_top_plane_points"
    assert result.timestamp_s == 9.5
    assert result.diagnostics["source_timestamp_ms"] == 2_500.0
    assert result.diagnostics["source_frame_id"] == 9
    assert result.diagnostics["source_reasons"] == ["insufficient_top_plane_points"]
    assert result.diagnostics["estimator"]["selected_points"] == 7


def test_frame_facade_calls_injected_estimator_exactly_once_with_frame_inputs() -> None:
    identity = np.eye(4, dtype=np.float64)
    calibration = Calibration(
        state=CalibrationState.BASE_VALIDATED,
        table_plane=Plane(normal=np.array([0.0, 0.0, 1.0]), d=0.0, frame="base"),
        T_base_from_head=identity,
        T_head_from_depth=identity,
    )
    expected_estimate = PoseEstimate(
        timestamp_ms=4_321.0,
        frame_id=51,
        frame="base",
        center_depth_m=(0.740, 0.015, 0.320),
        yaw_rad=math.pi / 2.0,
        calibration_state=CalibrationState.BASE_VALIDATED,
        base_registration="validated",
        geometry_valid=True,
        full_pose_valid=True,
        base_registration_valid=True,
        absolute_valid=True,
    )

    class EstimatorSpy:
        def __init__(self) -> None:
            self.calibration = calibration
            self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def estimate(self, *args: object, **kwargs: object) -> PoseEstimate:
            self.calls.append((args, kwargs))
            return expected_estimate

    estimator = EstimatorSpy()
    rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    depth = np.zeros((3, 4), dtype=np.uint16)
    support_mask = np.ones((3, 4), dtype=np.bool_)

    result = perceive_box_pose(
        rgb,
        depth,
        estimator=estimator,
        depth_scale=0.001,
        sensor_timestamp_ms=4_321.0,
        frame_id=51,
        timestamp_s=77.25,
        rgb_support_mask=support_mask,
    )

    assert result.valid is True
    assert len(estimator.calls) == 1
    args, kwargs = estimator.calls[0]
    assert len(args) == 1
    assert args[0] is depth
    assert set(kwargs) == {
        "depth_scale",
        "timestamp_ms",
        "frame_id",
        "rgb_support_mask",
    }
    assert kwargs["depth_scale"] == 0.001
    assert kwargs["timestamp_ms"] == 4_321.0
    assert kwargs["frame_id"] == 51
    assert kwargs["rgb_support_mask"] is support_mask
    assert result.timestamp_s == 77.25
    assert result.diagnostics["source_timestamp_ms"] == 4_321.0
    assert result.diagnostics["source_frame_id"] == 51
    assert result.diagnostics["capture"]["rgb"] == {
        "provided": True,
        "shape": [3, 4, 3],
        "dtype": "uint8",
    }
    assert result.diagnostics["capture"]["depth_scale_m"] == 0.001
    assert result.diagnostics["capture"]["estimator_reused"] is True


def test_frame_facade_propagates_invalid_estimate_reason_and_provenance() -> None:
    calibration = Calibration()

    class EstimatorSpy:
        def __init__(self) -> None:
            self.calls = 0

        def estimate(self, *args: object, **kwargs: object) -> PoseEstimate:
            self.calls += 1
            return PoseEstimate(
                timestamp_ms=900.0,
                frame_id=6,
                reasons=("insufficient_top_plane_points",),
                diagnostics={"selected_points": 5},
            )

    estimator = EstimatorSpy()
    result = perceive_box_pose(
        object(),
        object(),
        calibration=calibration,
        estimator=estimator,
        depth_scale=None,
        sensor_timestamp_ms=900.0,
        frame_id=6,
        timestamp_s=10.0,
    )

    assert estimator.calls == 1
    assert result.valid is False
    assert result.reason == "insufficient_top_plane_points"
    assert result.timestamp_s == 10.0
    assert result.diagnostics["source_timestamp_ms"] == 900.0
    assert result.diagnostics["source_frame_id"] == 6
    assert result.diagnostics["estimator"]["selected_points"] == 5


def test_frame_facade_can_construct_existing_estimator_from_explicit_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intrinsics = CameraIntrinsics(
        width=4,
        height=3,
        fx=100.0,
        fy=100.0,
        cx=1.5,
        cy=1.0,
    )
    calibration = Calibration()
    construction_calls: list[tuple[object, object, object]] = []
    estimate_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class ConstructedEstimator:
        def estimate(self, *args: object, **kwargs: object) -> PoseEstimate:
            estimate_calls.append((args, kwargs))
            return PoseEstimate(
                timestamp_ms=125.0,
                frame_id=2,
                reasons=("table_plane_missing",),
            )

    def construct(
        selected_intrinsics: object,
        selected_calibration: object,
        selected_config: object,
    ) -> ConstructedEstimator:
        construction_calls.append(
            (selected_intrinsics, selected_calibration, selected_config)
        )
        return ConstructedEstimator()

    monkeypatch.setattr(box_perception, "ParcelPoseEstimator", construct)

    result = perceive_box_pose(
        object(),
        object(),
        intrinsics,
        calibration,
        depth_scale=0.001,
        sensor_timestamp_ms=125.0,
        frame_id=2,
        timestamp_s=5.0,
    )

    assert construction_calls == [(intrinsics, calibration, None)]
    assert len(estimate_calls) == 1
    assert result.valid is False
    assert result.reason == "table_plane_missing"
    assert result.diagnostics["capture"]["estimator_reused"] is False
