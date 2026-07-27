import math

import pytest

from parcel_pose.angles import (
    classify_canonical_angle_deg,
    line_angle_difference_deg,
    line_angle_difference_rad,
    normalize_line_angle_deg,
    normalize_signed_line_angle_deg,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-360.0, 0.0),
        (-181.0, 179.0),
        (-180.0, 0.0),
        (-1.0, 179.0),
        (0.0, 0.0),
        (179.0, 179.0),
        (180.0, 0.0),
        (540.0, 0.0),
    ],
)
def test_normalize_line_angle_deg(value, expected):
    assert normalize_line_angle_deg(value) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(-91.0, 89.0), (-90.0, -90.0), (-1.0, -1.0), (90.0, -90.0), (179.0, -1.0)],
)
def test_normalize_signed_line_angle_deg(value, expected):
    assert normalize_signed_line_angle_deg(value) == pytest.approx(expected)


def test_line_difference_is_invariant_to_half_turns():
    baseline = line_angle_difference_deg(179.0, 2.0)
    assert baseline == pytest.approx(-3.0)
    assert line_angle_difference_deg(179.0 + 180.0, 2.0) == pytest.approx(baseline)
    assert line_angle_difference_rad(math.radians(179.0), math.radians(2.0)) == pytest.approx(
        math.radians(-3.0)
    )


@pytest.mark.parametrize(
    ("angle", "reference", "residual"),
    [
        (0.0, 0, 0.0),
        (44.999, 0, 44.999),
        (45.001, 90, -44.999),
        (90.0, 90, 0.0),
        (134.999, 90, 44.999),
        (135.001, 0, -44.999),
        (179.0, 0, -1.0),
    ],
)
def test_canonical_regions(angle, reference, residual):
    result = classify_canonical_angle_deg(angle)
    assert result.status == "constrained"
    assert result.reference_deg == reference
    assert result.residual_deg == pytest.approx(residual)


@pytest.mark.parametrize("angle", [45.0, 135.0, -45.0])
def test_exact_canonical_boundary_abstains(angle):
    result = classify_canonical_angle_deg(angle)
    assert result.status == "reference_ambiguous"
    assert result.reference_deg is None
    assert result.residual_deg is None


def test_uncertainty_touching_boundary_abstains():
    result = classify_canonical_angle_deg(42.0, uncertainty_deg=3.0)
    assert result.status == "reference_ambiguous"
    assert result.classification_margin_deg == pytest.approx(3.0)


def test_invalid_long_short_assignment_has_no_class():
    result = classify_canonical_angle_deg(10.0, long_short_assignment_valid=False)
    assert result.status == "axis_90_ambiguous"
    assert result.reference_deg is None
