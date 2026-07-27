import json

import numpy as np
import pytest

from parcel_pose.models import CalibrationState, PoseEstimate
from parcel_pose.output import (
    UnsafeOutputError,
    dumps_strict,
    pose_estimate_to_dict,
    validate_perception_only_keys,
)


@pytest.mark.parametrize(
    "key",
    [
        "robot_address",
        "recommended_command",
        "servo_velocity",
        "power_on",
        "contact_action",
        "arm_trajectory",
        "grasp_width",
        "target_end_effector_pose",
        "tcp_target",
    ],
)
def test_prohibited_nested_output_keys_are_rejected(key):
    payload = {"safe": [{"nested": {key: 1}}]}
    with pytest.raises(UnsafeOutputError, match="prohibited output key"):
        dumps_strict(payload)


def test_strict_json_rejects_nonfinite_values():
    with pytest.raises(ValueError, match="non-finite"):
        dumps_strict({"diagnostics": {"score": np.nan}})
    with pytest.raises(ValueError, match="non-finite"):
        dumps_strict({"diagnostics": {"score": np.inf}})


def test_missing_base_registration_strips_all_coordinate_base_fields():
    estimate = PoseEstimate(
        center_base_xy_m=(0.5, -0.1),
        top_center_base_xyz_m=(0.5, -0.1, 0.9),
        box_center_base_xyz_m=(0.5, -0.1, 0.825),
        long_axis_base_xy=(1.0, 0.0),
        yaw_mod_180_deg=12.0,
        geometry_valid=True,
        full_pose_valid=True,
        base_registration_valid=False,
        reasons=("missing_transform",),
    )
    output = pose_estimate_to_dict(estimate)
    assert "center_base_xy_m" not in output
    assert "top_center_base_xyz_m" not in output
    assert "box_center_base_xyz_m" not in output
    assert "long_axis_base_xy" not in output
    assert "long_axis_yaw_base_deg" not in output
    assert output["confidence"]["reasons"] == ["missing_transform"]


def test_base_fields_and_absolute_validity_are_independent():
    estimate = PoseEstimate(
        center_base_xy_m=(0.5, -0.1),
        top_center_base_xyz_m=(0.5, -0.1, 0.9),
        box_center_base_xyz_m=(0.5, -0.1, 0.825),
        long_axis_base_xy=(1.0, 0.0),
        short_axis_base_xy=(0.0, 1.0),
        yaw_mod_180_deg=12.0,
        geometry_valid=True,
        full_pose_valid=True,
        base_registration_valid=True,
        absolute_valid=False,
        calibration_state=CalibrationState.PLANE_CALIBRATED_PARTIAL,
        base_registration="nominal_unverified",
    )
    output = pose_estimate_to_dict(estimate)
    assert output["center_base_xy_m"] == [0.5, -0.1]
    assert output["top_center_base_xyz_m"] == [0.5, -0.1, 0.9]
    assert output["box_center_base_xyz_m"] == [0.5, -0.1, 0.825]
    assert not output["confidence"]["absolute_base_pose_valid"]
    validate_perception_only_keys(output)
    json.dumps(output, allow_nan=False)


def test_absolute_valid_requires_base_validated_full_pose():
    with pytest.raises(ValueError, match="absolute_valid requires"):
        PoseEstimate(absolute_valid=True)


def test_base_validated_label_without_transform_registration_is_not_absolute():
    estimate = PoseEstimate(
        calibration_state=CalibrationState.BASE_VALIDATED,
        base_registration="unavailable",
        base_registration_valid=False,
    )

    output = pose_estimate_to_dict(estimate)

    assert output["calibration"]["absolute_base_validated"] is False
    assert output["confidence"]["absolute_base_pose_valid"] is False
