from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from parcel_pose.calibration import load_calibration, load_json
from parcel_pose.estimator import EstimationEvidence, ParcelPoseEstimator
from parcel_pose.evaluation import base_pose_from_estimate
from parcel_pose.models import EstimatorConfig
from parcel_pose.projection import unproject_plane_points
from parcel_pose.recording import SessionReader
from parcel_pose.visualization import project_points_to_pixels


EXPECTED_MANIFEST_SHA256 = (
    "7362b992c87b80fd5f66b6ac52999dd854f39b6b80c7b767d70c70c1d272bd2c"
)
EXPECTED_CALIBRATION_SHA256 = (
    "c154775edb0e4992fc401fc1653651ceffdd3a3535e2f676a643c0b211978453"
)
CRITICAL_SOURCE_INDEX = 423


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _line_delta_deg(left: float, right: float) -> float:
    delta = abs((float(left) - float(right)) % 180.0)
    return min(delta, 180.0 - delta)


def _component_polygon_iou(
    evidence: EstimationEvidence,
    intrinsics,
) -> float:
    cv2 = pytest.importorskip("cv2")
    assert evidence.rectangle.corners_xy_m is not None
    number, labels, stats, _ = cv2.connectedComponentsWithStats(
        evidence.projection.mask.astype(np.uint8),
        connectivity=8,
    )
    assert number > 1
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = labels == label

    corners_depth = unproject_plane_points(
        evidence.rectangle.corners_xy_m,
        evidence.projection.plane,
        origin=evidence.projection.origin_3d_m,
        basis=(
            evidence.projection.basis_u_3d,
            evidence.projection.basis_v_3d,
        ),
    )
    pixels = project_points_to_pixels(corners_depth, intrinsics)
    polygon = np.zeros_like(component, dtype=np.uint8)
    cv2.fillPoly(polygon, [np.rint(pixels).astype(np.int32)], 1)
    polygon_mask = polygon.astype(np.bool_)
    intersection = int(np.count_nonzero(component & polygon_mask))
    union = int(np.count_nonzero(component | polygon_mask))
    return intersection / union


def test_measured_prior_meets_box_complex_continuity_and_overlay_gate() -> None:
    palletizing_root = Path(__file__).resolve().parents[2]
    session_path = palletizing_root / "recordings_" / "codex_640x480" / "box_complex"
    calibration_path = (
        palletizing_root / "out" / "box_complex_eval" / "calibration_fk_nominal.json"
    )
    if not session_path.is_dir() or not calibration_path.is_file():
        pytest.skip("local box_complex acceptance fixture is unavailable")

    manifest_path = session_path / "manifest.json"
    assert _sha256(manifest_path) == EXPECTED_MANIFEST_SHA256
    assert _sha256(calibration_path) == EXPECTED_CALIBRATION_SHA256

    config_path = palletizing_root / "Codex" / "configs" / "d435_rby1_nominal.json"
    config = EstimatorConfig.from_root_config(load_json(config_path))
    calibration = load_calibration(calibration_path)
    reader = SessionReader(session_path)
    estimator = ParcelPoseEstimator(
        reader.metadata.depth_profile.intrinsics,
        calibration,
        config,
    )

    valid: list[bool] = []
    centers: list[np.ndarray | None] = []
    yaws: list[float | None] = []
    confidences: list[float] = []
    critical_evidence: EstimationEvidence | None = None
    critical_frame_id: int | None = None
    for index, frame in enumerate(reader):
        estimate = estimator.estimate(
            frame.raw_depth_z16,
            depth_scale=reader.metadata.depth_scale_m,
            timestamp_ms=frame.depth_timestamp_ms,
            frame_id=frame.depth_frame_number,
        )
        pose = base_pose_from_estimate(estimate, calibration)
        valid.append(pose is not None)
        centers.append(
            None
            if pose is None
            else np.asarray(pose.box_center_xyz_m, dtype=np.float64)
        )
        yaws.append(None if pose is None else float(pose.yaw_signed_deg))
        confidences.append(float(estimate.per_field_confidence.get("yaw", 0.0)))
        if index == CRITICAL_SOURCE_INDEX:
            critical_evidence = estimator.last_evidence
            critical_frame_id = estimate.frame_id

    center_steps_mm: list[float] = []
    yaw_steps_deg: list[float] = []
    for index in range(1, len(valid)):
        if not (valid[index - 1] and valid[index]):
            continue
        center_steps_mm.append(
            1000.0
            * float(np.linalg.norm(centers[index] - centers[index - 1]))
        )
        yaw_steps_deg.append(_line_delta_deg(yaws[index], yaws[index - 1]))

    assert len(valid) == 547
    assert sum(valid) >= 535
    assert float(np.percentile(center_steps_mm, 95)) <= 22.0
    assert max(center_steps_mm) < 50.0
    assert float(np.percentile(yaw_steps_deg, 95)) <= 5.6
    assert max(yaw_steps_deg) < 40.0
    assert sum(step > 45.0 for step in yaw_steps_deg) == 0
    assert not [
        index
        for index, (is_valid, confidence) in enumerate(zip(valid, confidences, strict=True))
        if is_valid and confidence < 0.25
    ]

    assert all(valid[418:427])
    assert critical_frame_id == 1363
    assert confidences[CRITICAL_SOURCE_INDEX] >= 0.90
    assert critical_evidence is not None
    assert (
        _component_polygon_iou(
            critical_evidence,
            reader.metadata.depth_profile.intrinsics,
        )
        >= 0.90
    )
