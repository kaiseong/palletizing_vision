from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from parcel_pose.calibration import (
    calibrate_table_plane_from_session,
    fit_empty_table_plane_result,
    load_calibration,
    load_json,
    save_calibration,
)
from parcel_pose.models import CalibrationState
from parcel_pose.recording import write_session
from parcel_pose.session import RecordedFrame, SessionValidationError

from .synthetic_scene import noisy_plane_points
from .test_recording import make_metadata


def test_empty_table_ransac_rejects_non_table_planes_and_reports_evidence() -> None:
    points, truth_normal, truth_d, _ = noisy_plane_points(
        inlier_count=1_000,
        outlier_count=240,
        noise_std_m=0.0008,
        seed=808,
    )

    result = fit_empty_table_plane_result(
        points,
        tolerance_m=0.004,
        iterations=400,
        min_inlier_ratio=0.50,
        seed=91,
    )

    assert float(result.plane.normal @ truth_normal) > 0.999
    assert abs(result.plane.d - truth_d) < 0.002
    assert result.inlier_ratio > 0.75
    assert result.inlier_rms_m < 0.002


def _flat_table_frame(index: int, *, width: int = 64, height: int = 48) -> RecordedFrame:
    rng = np.random.default_rng(500 + index)
    depth = np.full((height, width), 800, dtype=np.uint16)
    outliers = rng.random((height, width)) < 0.02
    depth[outliers] = 1_150
    color = np.full((height, width, 3), 20 + index, dtype=np.uint8)
    return RecordedFrame(
        raw_depth_z16=depth,
        raw_color_bgr=color,
        depth_timestamp_ms=1_000.0 + 33.3 * index,
        color_timestamp_ms=1_000.5 + 33.3 * index,
        depth_frame_number=100 + index,
        color_frame_number=200 + index,
    )


def test_session_calibration_persists_global_and_per_frame_quality(tmp_path: Path) -> None:
    session = tmp_path / "empty_table"
    artifact = tmp_path / "table_plane.json"
    frames = [_flat_table_frame(index) for index in range(4)]
    write_session(session, make_metadata(width=64, height=48), frames)
    config_path = Path(__file__).resolve().parents[1] / "configs" / "d435_rby1_nominal.json"

    calibration = calibrate_table_plane_from_session(
        session,
        load_json(config_path),
        stride=2,
    )
    save_calibration(artifact, calibration)
    restored = load_calibration(artifact)

    assert restored.state is CalibrationState.PLANE_CALIBRATED_PARTIAL
    assert restored.table_plane is not None
    assert restored.table_plane.normal[2] < -0.999
    np.testing.assert_allclose(restored.table_plane.d, -0.8, atol=0.002)
    fit = restored.diagnostics["table_plane_fit"]
    assert fit["inlier_ratio"] > 0.95
    assert fit["normal_faces_camera"] is True
    assert fit["quality_passed"] is True
    assert len(fit["per_frame"]) == 4
    assert all(item["inlier_ratio"] > 0.95 for item in fit["per_frame"])
    assert set(fit["quality_checks"]) == {
        "global_inlier_ratio",
        "global_inlier_rms_m",
        "minimum_frame_inlier_ratio",
        "maximum_frame_p95_residual_m",
    }


def test_session_calibration_rejects_inconsistent_table_planes(tmp_path: Path) -> None:
    session = tmp_path / "inconsistent_empty_table"
    frames = []
    for index, depth_mm in enumerate((800, 1_000)):
        frames.append(
            RecordedFrame(
                raw_depth_z16=np.full((48, 64), depth_mm, dtype=np.uint16),
                raw_color_bgr=np.full((48, 64, 3), 20 + index, dtype=np.uint8),
                depth_timestamp_ms=1_000.0 + 33.3 * index,
                color_timestamp_ms=1_000.5 + 33.3 * index,
                depth_frame_number=100 + index,
                color_frame_number=200 + index,
            )
        )
    write_session(session, make_metadata(width=64, height=48), frames)
    config_path = Path(__file__).resolve().parents[1] / "configs" / "d435_rby1_nominal.json"

    with pytest.raises(SessionValidationError, match="quality gate failed"):
        calibrate_table_plane_from_session(
            session,
            load_json(config_path),
            stride=2,
        )
