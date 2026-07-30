"""Line-angle conventions for a rectangle with 180-degree symmetry."""

from __future__ import annotations

from dataclasses import dataclass
import math


def normalize_line_angle_rad(angle_rad: float) -> float:
    """Normalize an unoriented line angle to ``[0, pi)``."""

    return float(angle_rad) % math.pi


def normalize_signed_line_angle_rad(angle_rad: float) -> float:
    """Normalize an unoriented line angle to ``[-pi/2, pi/2)``."""

    return (float(angle_rad) + math.pi / 2.0) % math.pi - math.pi / 2.0


def line_angle_difference_rad(angle_rad: float, reference_rad: float) -> float:
    """Signed shortest difference between two unoriented lines."""

    return normalize_signed_line_angle_rad(float(angle_rad) - float(reference_rad))


def normalize_line_angle_deg(angle_deg: float) -> float:
    return float(angle_deg) % 180.0


def normalize_signed_line_angle_deg(angle_deg: float) -> float:
    return (float(angle_deg) + 90.0) % 180.0 - 90.0


def line_angle_difference_deg(angle_deg: float, reference_deg: float) -> float:
    return normalize_signed_line_angle_deg(float(angle_deg) - float(reference_deg))


@dataclass(frozen=True, slots=True)
class CanonicalAngleResult:
    reference_deg: int | None
    residual_deg: float | None
    classification_margin_deg: float
    status: str


def classify_canonical_angle_deg(
    angle_deg: float | None,
    *,
    uncertainty_deg: float = 0.0,
    long_short_assignment_valid: bool = True,
    boundary_tolerance_deg: float = 1e-9,
) -> CanonicalAngleResult:
    """Classify a long-axis line as the 0- or 90-degree family.

    The exact 45/135 degree boundaries, or an uncertainty interval touching a
    boundary, abstain with ``reference_ambiguous``.
    """

    if angle_deg is None or not long_short_assignment_valid:
        return CanonicalAngleResult(None, None, 0.0, "axis_90_ambiguous")
    uncertainty = float(uncertainty_deg)
    if not math.isfinite(uncertainty) or uncertainty < 0.0:
        raise ValueError("uncertainty_deg must be finite and non-negative")
    theta = normalize_line_angle_deg(angle_deg)
    margin = min(abs(theta - 45.0), abs(theta - 135.0))
    if margin <= uncertainty + float(boundary_tolerance_deg):
        return CanonicalAngleResult(None, None, max(0.0, margin), "reference_ambiguous")
    reference = 0 if theta < 45.0 or theta > 135.0 else 90
    residual = line_angle_difference_deg(theta, float(reference))
    return CanonicalAngleResult(reference, residual, margin, "constrained")



__all__ = [
    "CanonicalAngleResult",
    "classify_canonical_angle_deg",
    "line_angle_difference_deg",
    "line_angle_difference_rad",
    "normalize_line_angle_deg",
    "normalize_line_angle_rad",
    "normalize_signed_line_angle_deg",
    "normalize_signed_line_angle_rad",
]
