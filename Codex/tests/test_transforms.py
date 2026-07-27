import json
from pathlib import Path

import numpy as np
import pytest

from parcel_pose.calibration import nominal_calibration_from_config
from parcel_pose.models import Calibration
from parcel_pose.transforms import (
    compose_base_from_depth,
    invert_transform,
    rotation_from_euler_zyx,
    transform_directions,
    transform_from_euler_zyx,
    transform_points,
)
from parcel_pose.calibration import factory_extrinsics_to_transform
from parcel_pose.session import FactoryExtrinsics


def test_nominal_rz_ry_rx_axis_mapping():
    rotation = rotation_from_euler_zyx(-90.0, 0.0, -90.0, degrees=True)
    np.testing.assert_allclose(rotation @ [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(rotation @ [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(rotation @ [0.0, 1.0, 0.0], [0.0, 0.0, -1.0], atol=1e-12)


def test_transform_inverse_roundtrip_points_and_directions():
    transform = transform_from_euler_zyx([0.2, -0.1, 0.4], 0.2, -0.3, 0.5)
    points = np.array([[0.1, 0.2, 0.3], [-0.4, 0.5, 1.2]])
    directions = np.array([[1.0, 0.0, 0.0], [0.0, 0.3, -0.8]])
    inverse = invert_transform(transform)
    np.testing.assert_allclose(transform_points(transform_points(points, transform), inverse), points)
    np.testing.assert_allclose(
        transform_directions(transform_directions(directions, transform), inverse),
        directions,
        atol=1e-12,
    )


def test_noncommuting_target_from_source_chain():
    base_from_head = transform_from_euler_zyx([0.2, 0.1, 0.3], 0.1, 0.2, 0.3)
    head_from_color = transform_from_euler_zyx([0.05, -0.01, 0.06], -0.4, 0.1, -0.2)
    color_from_depth = transform_from_euler_zyx([-0.015, 0.002, 0.001], 0.01, -0.03, 0.02)
    expected = base_from_head @ head_from_color @ color_from_depth
    actual = compose_base_from_depth(base_from_head, head_from_color, color_from_depth)
    np.testing.assert_allclose(actual, expected)
    assert not np.allclose(actual, color_from_depth @ head_from_color @ base_from_head)
    calibration = Calibration(
        T_base_from_head=base_from_head,
        T_head_from_color=head_from_color,
        E_color_from_depth=color_from_depth,
    )
    np.testing.assert_allclose(calibration.T_base_from_depth, expected)


@pytest.mark.parametrize(
    "values",
    [
        (None, np.eye(4), np.eye(4)),
        (np.eye(4), None, np.eye(4)),
        (np.eye(4), np.eye(4), None),
    ],
)
def test_missing_chain_member_prevents_base_transform(values):
    assert compose_base_from_depth(*values) is None


def test_direct_head_from_depth_chain_is_supported():
    base_from_head = transform_from_euler_zyx([0.1, 0.2, 0.3], 0.2, 0.0, 0.1)
    head_from_depth = transform_from_euler_zyx([0.04, 0.0, 0.07], -0.5, 0.0, -0.4)
    result = compose_base_from_depth(
        base_from_head, None, None, T_head_from_depth=head_from_depth
    )
    np.testing.assert_allclose(result, base_from_head @ head_from_depth)


def test_principal_point_and_front_glass_are_not_transform_offsets():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs" / "d435_rby1_nominal.json").read_text())
    calibration = nominal_calibration_from_config(config, E_color_from_depth=np.eye(4))
    np.testing.assert_allclose(
        calibration.T_head_from_color[:3, 3], [0.049, -0.0115, 0.057]
    )
    assert -0.0042 not in calibration.T_head_from_color[:3, 3]
    assert "cx" not in config["calibration"]["nominal_T_head_from_color"]
    assert config["camera"]["front_glass_depth_start_usage"].startswith("mechanical_datum_only")


def test_realsense_factory_rotation_is_decoded_from_column_major_storage():
    rotation = rotation_from_euler_zyx(0.21, -0.34, 0.57)
    translation = np.array([0.012, -0.004, 0.003])
    extrinsics = FactoryExtrinsics(
        target_stream="color",
        source_stream="depth",
        rotation=tuple(rotation.reshape(-1, order="F")),
        translation_m=tuple(translation),
    )

    transform = factory_extrinsics_to_transform(extrinsics)

    np.testing.assert_allclose(transform[:3, :3], rotation, atol=1e-12)
    np.testing.assert_allclose(transform[:3, 3], translation, atol=1e-12)
    assert not np.allclose(transform[:3, :3], rotation.T)
