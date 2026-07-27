from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from parcel_pose.evaluation import (
    _allocate_temporary_video,
    _timing_summary,
    base_pose_from_estimate,
    draw_evaluation_overlay,
    evaluate_session_video,
)
from parcel_pose.models import (
    Calibration,
    CalibrationState,
    CameraIntrinsics,
    EstimatorConfig,
    Plane,
    PoseEstimate,
)


def _nominal_calibration() -> Calibration:
    return Calibration(
        state=CalibrationState.PLANE_CALIBRATED_PARTIAL,
        table_plane=Plane(normal=[0.0, 0.0, 1.0], d=0.75, frame="depth"),
        T_base_from_head=np.eye(4),
        T_head_from_depth=np.eye(4),
    )


def _valid_estimate() -> PoseEstimate:
    return PoseEstimate(
        frame="table_plane",
        center_depth_m=(0.70, -0.05, 0.90),
        yaw_rad=math.radians(20.0),
        yaw_mod_180_deg=20.0,
        observability={"yaw": "constrained"},
        per_field_confidence={"yaw": 0.8},
        diagnostics={"nominal_unverified_base": {"yaw_rad": math.radians(170.0)}},
        geometry_valid=True,
        full_pose_valid=True,
        calibration_state=CalibrationState.PLANE_CALIBRATED_PARTIAL,
        base_registration="nominal_unverified",
    )


def test_base_pose_reports_volume_center_and_nominal_registration() -> None:
    pose = base_pose_from_estimate(_valid_estimate(), _nominal_calibration())

    assert pose is not None
    np.testing.assert_allclose(pose.top_center_xyz_m, [0.70, -0.05, 0.90])
    np.testing.assert_allclose(pose.box_center_xyz_m, [0.70, -0.05, 0.825])
    assert pose.yaw_mod_180_deg == pytest.approx(170.0)
    assert pose.yaw_signed_deg == pytest.approx(-10.0)
    assert pose.canonical_reference_deg == 0
    assert pose.canonical_residual_deg == pytest.approx(-10.0)
    assert pose.registration == "nominal_unverified"


def test_base_pose_fails_closed_without_transform_chain() -> None:
    calibration = Calibration(
        state=CalibrationState.PLANE_CALIBRATED_PARTIAL,
        table_plane=Plane(normal=[0.0, 0.0, 1.0], d=0.75, frame="depth"),
    )

    assert base_pose_from_estimate(_valid_estimate(), calibration) is None


def test_recorded_timing_preserves_total_duration_not_only_median_interval() -> None:
    timing = _timing_summary([0.0, 100.0, 200.0, 400.0], nominal_fps=30.0)

    assert timing["recorded_duration_sec"] == pytest.approx(0.4)
    assert timing["effective_stored_fps"] == pytest.approx(7.5)
    assert timing["median_arrival_fps"] == pytest.approx(10.0)


def test_evaluation_rejects_colliding_output_paths_before_session_read(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result.mp4"

    with pytest.raises(ValueError, match="output paths must be distinct"):
        evaluate_session_video(
            tmp_path / "missing_session",
            _nominal_calibration(),
            EstimatorConfig(),
            output,
            output_summary=output,
        )


def test_temporary_video_is_unique_and_preserves_legacy_name(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    output = tmp_path / "result.mp4"
    legacy = tmp_path / "result.tmp.mp4"
    legacy.write_bytes(b"user-owned")

    temporary = _allocate_temporary_video(output)
    try:
        assert temporary != legacy
        assert legacy.read_bytes() == b"user-owned"
        writer = cv2.VideoWriter(
            str(temporary),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            (32, 24),
        )
        assert writer.isOpened()
        writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
        writer.release()

        capture = cv2.VideoCapture(str(temporary))
        try:
            ok, frame = capture.read()
            assert ok
            assert frame.shape[:2] == (24, 32)
        finally:
            capture.release()
    finally:
        temporary.unlink(missing_ok=True)

    assert legacy.read_bytes() == b"user-owned"


def test_overlay_draws_base_pose_text_without_evidence() -> None:
    pytest.importorskip("cv2")
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    intrinsics = CameraIntrinsics(
        width=320,
        height=240,
        fx=300.0,
        fy=300.0,
        cx=159.5,
        cy=119.5,
    )
    estimate = _valid_estimate()
    pose = base_pose_from_estimate(estimate, _nominal_calibration())

    rendered = draw_evaluation_overlay(
        image,
        estimate,
        pose,
        evidence=None,
        color_from_depth=np.eye(4),
        color_intrinsics=intrinsics,
        frame_index=0,
        frame_count=1,
        estimator_latency_ms=12.5,
    )

    assert rendered.shape == image.shape
    assert rendered.dtype == np.uint8
    assert np.count_nonzero(rendered) > 0
