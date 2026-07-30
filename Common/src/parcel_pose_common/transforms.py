"""Rigid-transform helpers with target-from-source naming."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def rotation_from_euler_zyx(
    roll: float,
    pitch: float,
    yaw: float,
    *,
    degrees: bool = False,
) -> FloatArray:
    """Return ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``."""

    if degrees:
        roll, pitch, yaw = (math.radians(float(v)) for v in (roll, pitch, yaw))
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    rx = np.array(((1.0, 0.0, 0.0), (0.0, cr, -sr), (0.0, sr, cr)))
    ry = np.array(((cp, 0.0, sp), (0.0, 1.0, 0.0), (-sp, 0.0, cp)))
    rz = np.array(((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0)))
    return rz @ ry @ rx


def make_transform(rotation: Any, translation: Any) -> FloatArray:
    rotation_array = np.asarray(rotation, dtype=np.float64)
    translation_array = np.asarray(translation, dtype=np.float64)
    if rotation_array.shape != (3, 3) or translation_array.shape != (3,):
        raise ValueError("rotation and translation must have shapes (3,3) and (3,)")
    if not np.all(np.isfinite(rotation_array)) or not np.all(np.isfinite(translation_array)):
        raise ValueError("transform components must be finite")
    if not np.allclose(rotation_array.T @ rotation_array, np.eye(3), atol=1e-7):
        raise ValueError("rotation must be orthonormal")
    if np.linalg.det(rotation_array) < 0.999999:
        raise ValueError("rotation must be right-handed with determinant +1")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_array
    transform[:3, 3] = translation_array
    return transform


def transform_from_euler_zyx(
    translation: Any,
    roll: float,
    pitch: float,
    yaw: float,
    *,
    degrees: bool = False,
) -> FloatArray:
    return make_transform(
        rotation_from_euler_zyx(roll, pitch, yaw, degrees=degrees),
        translation,
    )


def validate_transform(transform: Any, *, name: str = "transform") -> FloatArray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7):
        raise ValueError(f"{name} rotation determinant is not +1")
    return matrix


def invert_transform(transform_target_from_source: Any) -> FloatArray:
    transform = validate_transform(transform_target_from_source)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return inverse


def transform_points(points_source: Any, transform_target_from_source: Any) -> FloatArray:
    points = np.asarray(points_source, dtype=np.float64)
    if points.shape[-1] != 3:
        raise ValueError("points must end in a length-3 coordinate")
    transform = validate_transform(transform_target_from_source)
    return points @ transform[:3, :3].T + transform[:3, 3]


def transform_directions(directions_source: Any, transform_target_from_source: Any) -> FloatArray:
    directions = np.asarray(directions_source, dtype=np.float64)
    if directions.shape[-1] != 3:
        raise ValueError("directions must end in a length-3 coordinate")
    transform = validate_transform(transform_target_from_source)
    return directions @ transform[:3, :3].T


def compose_base_from_depth(
    T_base_from_head: Any | None,
    T_head_from_color: Any | None,
    E_color_from_depth: Any | None,
    *,
    T_head_from_depth: Any | None = None,
) -> FloatArray | None:
    """Compose the canonical raw-depth chain, returning ``None`` if incomplete."""

    if T_base_from_head is None:
        return None
    base_from_head = validate_transform(T_base_from_head, name="T_base_from_head")
    if T_head_from_depth is not None:
        return base_from_head @ validate_transform(
            T_head_from_depth, name="T_head_from_depth"
        )
    if T_head_from_color is None or E_color_from_depth is None:
        return None
    return (
        base_from_head
        @ validate_transform(T_head_from_color, name="T_head_from_color")
        @ validate_transform(E_color_from_depth, name="E_color_from_depth")
    )


__all__ = [
    "compose_base_from_depth",
    "invert_transform",
    "make_transform",
    "rotation_from_euler_zyx",
    "transform_directions",
    "transform_from_euler_zyx",
    "transform_points",
    "validate_transform",
]
