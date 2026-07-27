from __future__ import annotations

import math

import numpy as np
import pytest

from parcel_pose.burst import BurstConfig, PoseBurstAggregator, aggregate_pose_burst
from parcel_pose.models import CalibrationState, PoseEstimate

from .synthetic_scene import line_angle_error_deg


def _pose(
    frame_id: int,
    *,
    timestamp_ms: float | None = None,
    center: tuple[float, float] | None = (0.10, -0.04),
    yaw_deg: float | None = 12.0,
    reasons: tuple[str, ...] = (),
    fresh: bool = True,
    center_long_state: str | None = None,
    center_short_state: str | None = None,
) -> PoseEstimate:
    yaw_rad = None if yaw_deg is None else math.radians(yaw_deg)
    long_axis = None if yaw_rad is None else (math.cos(yaw_rad), math.sin(yaw_rad))
    short_axis = None if yaw_rad is None else (-math.sin(yaw_rad), math.cos(yaw_rad))
    center_state = "both_edges" if center is not None else "underconstrained"
    return PoseEstimate(
        timestamp_ms=float(frame_id * 10 if timestamp_ms is None else timestamp_ms),
        frame_id=frame_id,
        frame="table_plane",
        center_plane_xy_m=center,
        yaw_rad=yaw_rad,
        yaw_mod_180_deg=None if yaw_deg is None else yaw_deg % 180.0,
        long_axis_plane_xy=long_axis,
        short_axis_plane_xy=short_axis,
        observability={
            "center_long": center_state if center_long_state is None else center_long_state,
            "center_short": center_state if center_short_state is None else center_short_state,
            "yaw": "constrained" if yaw_rad is not None else "underconstrained",
            "reference": "constrained" if yaw_rad is not None else "underconstrained",
        },
        per_field_confidence={
            "center_long": 0.9 if center is not None else 0.0,
            "center_short": 0.9 if center is not None else 0.0,
            "yaw": 0.9 if yaw_rad is not None else 0.0,
            "reference": 0.9 if yaw_rad is not None else 0.0,
        },
        diagnostics={"fresh": fresh},
        reasons=reasons,
        calibration_state=CalibrationState.PLANE_CALIBRATED_PARTIAL,
        geometry_valid=center is not None or yaw_rad is not None,
        full_pose_valid=center is not None and yaw_rad is not None,
    )


def test_line_angle_wrap_aggregates_179_0_1_near_zero_not_sixty() -> None:
    estimates = [
        _pose(1, yaw_deg=179.0, center=(0.100, -0.040)),
        _pose(2, yaw_deg=0.0, center=(0.101, -0.040)),
        _pose(3, yaw_deg=1.0, center=(0.099, -0.039)),
        _pose(4, yaw_deg=179.5, center=(0.100, -0.041)),
        _pose(5, yaw_deg=0.5, center=(0.100, -0.040)),
    ]

    aggregate = aggregate_pose_burst(estimates, min_valid_frames=5)

    assert aggregate.full_pose_valid, aggregate.reasons
    assert aggregate.yaw_rad is not None
    assert line_angle_error_deg(aggregate.yaw_rad, 0.0) < 0.25
    assert aggregate.yaw_mod_180_deg < 1.0 or aggregate.yaw_mod_180_deg > 179.0
    assert aggregate.canonical_reference_deg == 0
    assert abs(aggregate.canonical_residual_deg) < 0.25
    np.testing.assert_allclose(aggregate.center_plane_xy_m, [0.100, -0.040], atol=0.001)
    assert aggregate.diagnostics["burst"]["valid_yaw_frames"] == 5


def test_stale_duplicate_and_field_invalid_frames_do_not_vote_in_fresh_burst() -> None:
    estimates = [
        _pose(1, timestamp_ms=90.0, yaw_deg=80.0),  # before freshness barrier
        _pose(2, timestamp_ms=110.0, yaw_deg=9.0),
        _pose(3, timestamp_ms=120.0, yaw_deg=10.0),
        _pose(4, timestamp_ms=130.0, yaw_deg=11.0),
        _pose(4, timestamp_ms=130.0, yaw_deg=70.0),  # duplicate identity
        _pose(5, timestamp_ms=140.0, yaw_deg=75.0, reasons=("stale_frame",)),
        _pose(6, timestamp_ms=150.0, yaw_deg=None, center=None),
    ]

    aggregate = aggregate_pose_burst(
        estimates,
        min_valid_frames=3,
        min_timestamp_ms=100.0,
    )

    assert aggregate.full_pose_valid
    assert line_angle_error_deg(aggregate.yaw_rad, math.radians(10.0)) < 0.1
    assert aggregate.diagnostics["burst"]["input_frames"] == 7
    assert aggregate.diagnostics["burst"]["fresh_frames"] == 4
    assert aggregate.diagnostics["burst"]["valid_yaw_frames"] == 3
    assert aggregate.diagnostics["burst"]["valid_center_frames"] == 3


def test_center_and_yaw_jitter_abstain_with_explicit_temporal_reason() -> None:
    estimates = [
        _pose(1, center=(0.070, -0.040), yaw_deg=0.0),
        _pose(2, center=(0.085, -0.040), yaw_deg=10.0),
        _pose(3, center=(0.100, -0.040), yaw_deg=170.0),
        _pose(4, center=(0.115, -0.040), yaw_deg=9.0),
        _pose(5, center=(0.130, -0.040), yaw_deg=171.0),
    ]

    aggregate = aggregate_pose_burst(
        estimates,
        min_valid_frames=5,
        max_center_jitter_m=0.020,
        max_yaw_jitter_deg=4.0,
    )

    assert aggregate.full_pose_valid is False
    assert aggregate.center_plane_xy_m is None
    assert aggregate.yaw_rad is None
    assert "temporal_jitter_too_large" in aggregate.reasons
    assert "center_temporal_jitter_too_large" in aggregate.reasons
    assert "yaw_temporal_jitter_too_large" in aggregate.reasons
    assert aggregate.diagnostics["burst"]["center_jitter_m"] > 0.020
    assert aggregate.diagnostics["burst"]["yaw_jitter_deg"] > 4.0


@pytest.mark.parametrize("missing_field", ["center", "yaw"])
def test_per_field_invalidity_survives_aggregation(missing_field: str) -> None:
    estimates: list[PoseEstimate] = []
    for frame_id in range(1, 6):
        if missing_field == "center":
            estimates.append(
                _pose(
                    frame_id,
                    center=None,
                    yaw_deg=20.0 + 0.1 * frame_id,
                    center_long_state="both_edges",
                    center_short_state="underconstrained",
                )
            )
        else:
            estimates.append(_pose(frame_id, center=(0.10, -0.04), yaw_deg=None))

    aggregate = aggregate_pose_burst(estimates, min_valid_frames=5)

    assert aggregate.full_pose_valid is False
    if missing_field == "center":
        assert aggregate.yaw_rad is not None
        assert aggregate.center_plane_xy_m is None
        assert aggregate.observability["center_short"] == "underconstrained"
        assert "insufficient_valid_center_frames" in aggregate.reasons
    else:
        assert aggregate.center_plane_xy_m is not None
        assert aggregate.yaw_rad is None
        assert aggregate.observability["yaw"] == "underconstrained"
        assert "insufficient_valid_yaw_frames" in aggregate.reasons


def test_configured_aggregator_matches_functional_api() -> None:
    estimates = [_pose(index, yaw_deg=14.0 + 0.1 * index) for index in range(1, 6)]
    config = BurstConfig(
        min_valid_frames=5,
        max_center_jitter_m=0.010,
        max_yaw_jitter_deg=2.0,
    )

    class_result = PoseBurstAggregator(config).aggregate(estimates)
    function_result = aggregate_pose_burst(
        estimates,
        min_valid_frames=5,
        max_center_jitter_m=0.010,
        max_yaw_jitter_deg=2.0,
    )

    assert class_result == function_result


def test_burst_preserves_one_edge_inferred_crop_provenance() -> None:
    estimates = [
        _pose(
            frame_id,
            center=(0.10 + 0.0001 * frame_id, -0.04),
            yaw_deg=12.0,
            center_long_state="one_edge_inferred",
            center_short_state="both_edges",
        )
        for frame_id in range(1, 6)
    ]

    aggregate = aggregate_pose_burst(estimates, min_valid_frames=5)

    assert aggregate.full_pose_valid
    assert aggregate.observability["center_long"] == "one_edge_inferred"
    assert aggregate.observability["center_short"] == "both_edges"
