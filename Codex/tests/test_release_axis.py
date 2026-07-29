"""Opening-axis resolution for the slot-1 release.

The hands open along the torso-tip Y axis, so the reference direction is an
argument rather than a hard-coded base axis.  At the slot-1 ready pose the torso
yaw is zero, which makes torso Y and base Y coincide.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from parcel_pose.pallet_control import resolve_release_axis

from _factories import rotate_axis_about_z


LIMIT_RAD = math.radians(10.0)
BASE_Y = (0.0, 1.0, 0.0)


def test_measured_axis_on_negative_y_keeps_that_sign() -> None:
    axis, deviation = resolve_release_axis(
        (0.0, -1.0, 0.0), BASE_Y, max_deviation_rad=LIMIT_RAD
    )
    assert axis == pytest.approx((0.0, -1.0, 0.0))
    assert deviation == pytest.approx(0.0)


def test_measured_axis_on_positive_y_keeps_that_sign() -> None:
    axis, deviation = resolve_release_axis(
        (0.0, 1.0, 0.0), BASE_Y, max_deviation_rad=LIMIT_RAD
    )
    assert axis == pytest.approx((0.0, 1.0, 0.0))
    assert deviation == pytest.approx(0.0)


@pytest.mark.parametrize("degrees", [1.0, 5.0, 9.5, 20.0, 45.0])
def test_deviation_matches_the_rotation_and_axis_stays_on_reference(degrees) -> None:
    rotated = rotate_axis_about_z((0.0, -1.0, 0.0), degrees)
    axis, deviation = resolve_release_axis(rotated, BASE_Y, max_deviation_rad=LIMIT_RAD)
    assert axis == pytest.approx((0.0, -1.0, 0.0))
    assert deviation == pytest.approx(math.radians(degrees))
    # The commanded direction never picks up an X or Z component of its own.
    assert axis[0] == 0.0
    assert axis[2] == 0.0


def test_reference_axis_follows_a_rotated_torso() -> None:
    """A yawed torso opens along its own Y, not along base Y."""

    torso_y = rotate_axis_about_z(BASE_Y, 30.0)
    measured = rotate_axis_about_z((0.0, -1.0, 0.0), 30.0)
    axis, deviation = resolve_release_axis(
        measured, torso_y, max_deviation_rad=LIMIT_RAD
    )
    assert axis == pytest.approx(-np.asarray(torso_y))
    assert deviation == pytest.approx(0.0)


def test_both_axes_are_normalized() -> None:
    axis, deviation = resolve_release_axis(
        (0.0, -4.0, 0.0), (0.0, 7.0, 0.0), max_deviation_rad=LIMIT_RAD
    )
    assert axis == pytest.approx((0.0, -1.0, 0.0))
    assert deviation == pytest.approx(0.0)


def test_vertical_component_of_the_measured_axis_is_projected_out() -> None:
    axis, deviation = resolve_release_axis(
        (0.0, -1.0, 1.0), BASE_Y, max_deviation_rad=LIMIT_RAD
    )
    assert axis == pytest.approx((0.0, -1.0, 0.0))
    assert deviation == pytest.approx(math.radians(45.0))


def test_perpendicular_measured_axis_is_rejected() -> None:
    with pytest.raises(ValueError, match="no decidable sign"):
        resolve_release_axis((1.0, 0.0, 0.0), BASE_Y, max_deviation_rad=LIMIT_RAD)


def test_degenerate_and_nonfinite_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="inter_eef_axis_base must be a non-zero"):
        resolve_release_axis((0.0, 0.0, 0.0), BASE_Y, max_deviation_rad=LIMIT_RAD)
    with pytest.raises(ValueError, match="reference_axis_base must be a non-zero"):
        resolve_release_axis((0.0, -1.0, 0.0), (0.0, 0.0, 0.0),
                             max_deviation_rad=LIMIT_RAD)
    with pytest.raises(ValueError, match="must be finite"):
        resolve_release_axis((0.0, float("nan"), 0.0), BASE_Y,
                             max_deviation_rad=LIMIT_RAD)
    with pytest.raises(ValueError, match="max_deviation_rad"):
        resolve_release_axis((0.0, -1.0, 0.0), BASE_Y, max_deviation_rad=0.0)


def test_returned_axis_is_a_unit_float_array() -> None:
    axis, _ = resolve_release_axis((0.0, -1.0, 0.0), BASE_Y, max_deviation_rad=LIMIT_RAD)
    assert isinstance(axis, np.ndarray)
    assert axis.dtype == np.float64
    assert float(np.linalg.norm(axis)) == pytest.approx(1.0)
