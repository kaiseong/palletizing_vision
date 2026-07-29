"""Base-frame ``+/-Y`` release-axis resolution."""

from __future__ import annotations

import math

import numpy as np
import pytest

from parcel_pose.pallet_control import resolve_base_y_release_axis

from _factories import rotate_axis_about_z


LIMIT_RAD = math.radians(10.0)


def test_exact_negative_y_axis_has_no_deviation() -> None:
    axis, deviation = resolve_base_y_release_axis(
        (0.0, -1.0, 0.0), max_deviation_rad=LIMIT_RAD
    )
    assert axis == pytest.approx((0.0, -1.0, 0.0))
    assert deviation == pytest.approx(0.0)


def test_exact_positive_y_axis_keeps_its_sign() -> None:
    axis, deviation = resolve_base_y_release_axis(
        (0.0, 1.0, 0.0), max_deviation_rad=LIMIT_RAD
    )
    assert axis == pytest.approx((0.0, 1.0, 0.0))
    assert deviation == pytest.approx(0.0)


@pytest.mark.parametrize("degrees", [1.0, 5.0, 9.5, 20.0, 45.0])
def test_deviation_matches_the_rotation_and_axis_stays_on_y(degrees: float) -> None:
    rotated = rotate_axis_about_z((0.0, -1.0, 0.0), degrees)
    axis, deviation = resolve_base_y_release_axis(rotated, max_deviation_rad=LIMIT_RAD)
    assert axis == pytest.approx((0.0, -1.0, 0.0))
    assert deviation == pytest.approx(math.radians(degrees))
    # The opening direction never acquires an X or Z component.
    assert axis[0] == 0.0
    assert axis[2] == 0.0


def test_axis_is_normalized_before_resolution() -> None:
    axis, deviation = resolve_base_y_release_axis(
        (0.0, -4.0, 0.0), max_deviation_rad=LIMIT_RAD
    )
    assert axis == pytest.approx((0.0, -1.0, 0.0))
    assert deviation == pytest.approx(0.0)


def test_vertical_component_is_projected_out() -> None:
    axis, deviation = resolve_base_y_release_axis(
        (0.0, -1.0, 1.0), max_deviation_rad=LIMIT_RAD
    )
    assert axis == pytest.approx((0.0, -1.0, 0.0))
    assert deviation == pytest.approx(math.radians(45.0))


def test_undecidable_sign_is_rejected() -> None:
    with pytest.raises(ValueError, match="no decidable base-Y sign"):
        resolve_base_y_release_axis((1.0, 0.0, 0.0), max_deviation_rad=LIMIT_RAD)


def test_degenerate_and_nonfinite_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-zero direction"):
        resolve_base_y_release_axis((0.0, 0.0, 0.0), max_deviation_rad=LIMIT_RAD)
    with pytest.raises(ValueError, match="must be finite"):
        resolve_base_y_release_axis(
            (0.0, float("nan"), 0.0), max_deviation_rad=LIMIT_RAD
        )
    with pytest.raises(ValueError, match="max_deviation_rad"):
        resolve_base_y_release_axis((0.0, -1.0, 0.0), max_deviation_rad=0.0)


def test_returned_axis_is_a_plain_float_array() -> None:
    axis, _ = resolve_base_y_release_axis((0.0, -1.0, 0.0), max_deviation_rad=LIMIT_RAD)
    assert isinstance(axis, np.ndarray)
    assert axis.dtype == np.float64
    assert float(np.linalg.norm(axis)) == pytest.approx(1.0)
