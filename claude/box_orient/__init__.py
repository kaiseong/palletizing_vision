"""Colour-agnostic, depth-driven box orientation for RB-Y1 palletizing.

Public API:
    estimate_box_orientation(depth_m, intr, ...) -> BoxOrientation
    RecordingSource(session_dir) / RealsenseSource(...)  -> Frame(bgr, depth_m, intr, ...)
    draw_overlay(bgr, est, intr)
    camera_to_t5_static(camera)
"""

from __future__ import annotations

from .extrinsics import camera_to_t5_from_fk, camera_to_t5_static, is_calibrated
from .geometry import (
    CameraIntrinsics,
    SupportPlane,
    deproject_valid,
    fit_support_plane,
    plane_basis,
)
from .orientation import (
    BoxOrientation,
    OrientConfig,
    classify_0_90,
    discover_support_plane,
    estimate_box_orientation,
    yaw_about_normal,
)
from .segment import Segmentation, segment_box_top
from .sources import Frame, RecordingSource, RealsenseSource
from .viz import draw_overlay

__all__ = [
    "estimate_box_orientation",
    "BoxOrientation",
    "OrientConfig",
    "classify_0_90",
    "yaw_about_normal",
    "discover_support_plane",
    "segment_box_top",
    "Segmentation",
    "CameraIntrinsics",
    "SupportPlane",
    "deproject_valid",
    "fit_support_plane",
    "plane_basis",
    "RecordingSource",
    "RealsenseSource",
    "Frame",
    "draw_overlay",
    "camera_to_t5_static",
    "camera_to_t5_from_fk",
    "is_calibrated",
]
