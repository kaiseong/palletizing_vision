from __future__ import annotations

import math

import pytest

from parcel_pose_common.angles import line_angle_difference_rad
from parcel_pose_common.models import PoseResult


def test_valid_pose_result_has_explicit_units_and_base_line_yaw_semantics() -> None:
    source_diagnostics = {
        "source": "box_estimator",
        "registration": "validated",
        "quality": {"candidate_margin": 0.42},
    }

    result = PoseResult(
        x_m=0.740,
        y_m=-0.012,
        yaw_rad=math.radians(91.5),
        valid=True,
        reason="",
        timestamp_s=12.25,
        diagnostics=source_diagnostics,
    )

    assert result.x_m == pytest.approx(0.740)
    assert result.y_m == pytest.approx(-0.012)
    assert result.yaw_rad == pytest.approx(math.radians(91.5))
    assert result.timestamp_s == pytest.approx(12.25)
    assert result.frame == "base"
    assert line_angle_difference_rad(result.yaw_rad + math.pi, result.yaw_rad) == pytest.approx(
        0.0
    )
    assert result.diagnostics == source_diagnostics


def test_diagnostics_defensively_copy_estimator_provenance() -> None:
    source_diagnostics = {"source": "pallet_estimator", "frame_id": 17}
    result = PoseResult(
        x_m=0.865,
        y_m=0.139523,
        yaw_rad=-math.pi / 2.0,
        valid=True,
        reason="",
        timestamp_s=5.0,
        diagnostics=source_diagnostics,
    )

    source_diagnostics["frame_id"] = 99

    assert result.diagnostics == {"source": "pallet_estimator", "frame_id": 17}
    assert result.to_dict() == {
        "x_m": 0.865,
        "y_m": 0.139523,
        "yaw_rad": -math.pi / 2.0,
        "valid": True,
        "reason": "",
        "timestamp_s": 5.0,
        "frame": "base",
        "diagnostics": {"source": "pallet_estimator", "frame_id": 17},
    }


def test_invalid_pose_requires_reason_and_preserves_partial_pose_provenance() -> None:
    result = PoseResult(
        x_m=0.72,
        y_m=None,
        yaw_rad=None,
        valid=False,
        reason="base_registration_unavailable",
        timestamp_s=0.0,
        diagnostics={"source": "box_estimator", "frame_id": 4},
    )

    assert result.valid is False
    assert result.reason == "base_registration_unavailable"
    assert result.x_m == pytest.approx(0.72)
    assert result.y_m is None
    assert result.yaw_rad is None
    assert result.diagnostics["source"] == "box_estimator"

    with pytest.raises(ValueError, match="non-empty reason"):
        PoseResult(
            x_m=None,
            y_m=None,
            yaw_rad=None,
            valid=False,
            reason="   ",
            timestamp_s=1.0,
        )


@pytest.mark.parametrize("reason", [None, 17, False])
def test_pose_result_rejects_non_string_reason(reason: object) -> None:
    with pytest.raises(ValueError, match="reason must be a string"):
        PoseResult(
            x_m=None,
            y_m=None,
            yaw_rad=None,
            valid=False,
            reason=reason,  # type: ignore[arg-type]
            timestamp_s=1.0,
        )


@pytest.mark.parametrize("missing_field", ["x_m", "y_m", "yaw_rad"])
def test_valid_pose_requires_every_pose_component(missing_field: str) -> None:
    values: dict[str, float | None] = {
        "x_m": 0.7,
        "y_m": 0.1,
        "yaw_rad": math.pi / 2.0,
    }
    values[missing_field] = None

    with pytest.raises(ValueError, match="requires x_m, y_m, and yaw_rad"):
        PoseResult(
            **values,
            valid=True,
            reason="",
            timestamp_s=1.0,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"x_m": math.nan}, "x_m must be finite"),
        ({"y_m": math.inf}, "y_m must be finite"),
        ({"yaw_rad": -math.inf}, "yaw_rad must be finite"),
        ({"timestamp_s": math.nan}, "timestamp_s must be finite"),
        ({"timestamp_s": -0.001}, "timestamp_s must be non-negative"),
        ({"frame": "camera"}, "RB-Y1 'base' frame"),
        ({"valid": 1}, "valid must be a bool"),
        ({"diagnostics": []}, "diagnostics must be a mapping"),
    ],
)
def test_pose_result_rejects_ambiguous_or_non_finite_contract_values(
    overrides: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "x_m": 0.7,
        "y_m": 0.1,
        "yaw_rad": 0.0,
        "valid": True,
        "reason": "",
        "timestamp_s": 1.0,
        "diagnostics": {},
        "frame": "base",
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        PoseResult(**values)
