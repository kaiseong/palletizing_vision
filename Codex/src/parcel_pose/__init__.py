"""D435 parcel-pose perception package.

The package core deliberately has no import-time dependency on pyrealsense2.
"""

from .angles import (
    CanonicalAngleResult,
    classify_canonical_angle_deg,
    line_angle_difference_deg,
    line_angle_difference_rad,
    normalize_line_angle_deg,
    normalize_line_angle_rad,
    normalize_signed_line_angle_deg,
    normalize_signed_line_angle_rad,
)
from .models import (
    BoxDimensionPrior,
    BoxModel,
    Calibration,
    CalibrationState,
    CameraIntrinsics,
    EstimatorConfig,
    ObservabilityState,
    Plane,
    PoseEstimate,
)

__all__ = [
    "BoxDimensionPrior",
    "BoxModel",
    "Calibration",
    "CalibrationState",
    "CameraIntrinsics",
    "CanonicalAngleResult",
    "EstimatorConfig",
    "ObservabilityState",
    "Plane",
    "PoseEstimate",
    "classify_canonical_angle_deg",
    "line_angle_difference_deg",
    "line_angle_difference_rad",
    "normalize_line_angle_deg",
    "normalize_line_angle_rad",
    "normalize_signed_line_angle_deg",
    "normalize_signed_line_angle_rad",
]
