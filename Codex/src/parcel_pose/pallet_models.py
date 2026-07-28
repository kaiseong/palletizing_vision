"""Typed contracts for the pallet slot-1 hover perception primitive.

All lengths are metres, angles are radians, and rigid geometry is expressed in
the RB-Y1 base frame unless a field name says otherwise.  Invalid observations
carry ``None`` for unobservable geometry instead of manufacturing a pose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .models import Plane


FloatArray = NDArray[np.float64]


def _finite_positive(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _vector(
    value: ArrayLike | None,
    length: int,
    name: str,
) -> FloatArray | None:
    if value is None:
        return None
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite length-{length} vector")
    result = result.copy()
    result.setflags(write=False)
    return result


def _matrix(value: ArrayLike | None, columns: int, name: str) -> FloatArray | None:
    if value is None:
        return None
    result = np.asarray(value, dtype=np.float64)
    if (
        result.ndim != 2
        or result.shape[1] != columns
        or not np.all(np.isfinite(result))
    ):
        raise ValueError(f"{name} must be a finite Nx{columns} matrix")
    result = result.copy()
    result.setflags(write=False)
    return result


def _float_pair(values: Sequence[float], name: str) -> tuple[float, float]:
    result = tuple(_finite_positive(value, name) for value in values)
    if len(result) != 2:
        raise ValueError(f"{name} must contain two values")
    return result  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class PalletGeometry:
    """Fixed pinwheel-stack geometry measured by the operator."""

    outer_size_m: tuple[float, float] = (0.660, 0.658)
    opening_size_m: tuple[float, float] = (0.148, 0.149)
    slot1_offset_m: tuple[float, float] = (0.12800, 0.20175)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "outer_size_m", _float_pair(self.outer_size_m, "outer_size_m")
        )
        object.__setattr__(
            self, "opening_size_m", _float_pair(self.opening_size_m, "opening_size_m")
        )
        object.__setattr__(
            self, "slot1_offset_m", _float_pair(self.slot1_offset_m, "slot1_offset_m")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "outer_size_m": list(self.outer_size_m),
            "opening_size_m": list(self.opening_size_m),
            "slot1_offset_m": list(self.slot1_offset_m),
        }


@dataclass(frozen=True, slots=True)
class PalletPerceptionGates:
    """Physical observability gates; outer-rim evidence is intentionally optional."""

    min_inner_rim_count: int = 3
    max_opening_size_error_m: float = 0.015
    max_orthogonality_error_rad: float = math.radians(5.0)
    max_plane_p95_residual_m: float = 0.008
    max_center_spread_m: float = 0.008
    max_yaw_spread_rad: float = math.radians(2.0)
    max_start_yaw_residual_rad: float = math.radians(15.0)
    max_consecutive_center_jump_m: float = 0.030

    def __post_init__(self) -> None:
        if int(self.min_inner_rim_count) not in (3, 4):
            raise ValueError("min_inner_rim_count must be 3 or 4")
        object.__setattr__(self, "min_inner_rim_count", int(self.min_inner_rim_count))
        for name in (
            "max_opening_size_error_m",
            "max_orthogonality_error_rad",
            "max_plane_p95_residual_m",
            "max_center_spread_m",
            "max_yaw_spread_rad",
            "max_start_yaw_residual_rad",
            "max_consecutive_center_jump_m",
        ):
            object.__setattr__(self, name, _finite_positive(getattr(self, name), name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_inner_rim_count": self.min_inner_rim_count,
            "max_opening_size_error_m": self.max_opening_size_error_m,
            "max_orthogonality_error_deg": math.degrees(
                self.max_orthogonality_error_rad
            ),
            "max_plane_p95_residual_m": self.max_plane_p95_residual_m,
            "max_center_spread_m": self.max_center_spread_m,
            "max_yaw_spread_deg": math.degrees(self.max_yaw_spread_rad),
            "max_start_yaw_residual_deg": math.degrees(self.max_start_yaw_residual_rad),
            "max_consecutive_center_jump_m": self.max_consecutive_center_jump_m,
        }


@dataclass(frozen=True, slots=True)
class PalletEstimatorConfig:
    """Deterministic metric estimator settings for the fixed ready posture."""

    geometry: PalletGeometry = field(default_factory=PalletGeometry)
    gates: PalletPerceptionGates = field(default_factory=PalletPerceptionGates)
    workspace_x_m: tuple[float, float] = (0.15, 1.60)
    workspace_y_m: tuple[float, float] = (-0.90, 0.90)
    workspace_z_m: tuple[float, float] = (0.20, 1.10)
    stack_plane_z_m: tuple[float, float] = (0.25, 0.70)
    min_depth_m: float = 0.20
    max_depth_m: float = 1.50
    z_histogram_bin_m: float = 0.003
    plane_seed_band_m: float = 0.035
    plane_fit_tolerance_m: float = 0.008
    plane_slab_m: float = 0.012
    plane_fit_max_points: int = 30_000
    min_plane_points: int = 1_000
    grid_resolution_m: float = 0.002
    morphology_close_m: float = 0.010
    morphology_dilate_m: float = 0.004
    opening_component_min_m: float = 0.100
    opening_component_max_m: float = 0.220
    rim_outer_band_m: tuple[float, float] = (0.004, 0.024)
    min_rim_support_ratio: float = 0.25
    held_plane_min_separation_m: float = 0.080
    held_plane_max_uncertainty_m: float = 0.012
    rough_front_axis_base: tuple[float, float] = (1.0, 0.0)
    l_corner_edge_band_m: float = 0.012
    l_corner_min_front_support_m: float = 0.450
    l_corner_min_side_support_m: float = 0.180
    l_corner_max_line_p95_residual_m: float = 0.008
    l_corner_max_connection_gap_m: float = 0.015
    l_corner_max_orthogonality_error_rad: float = math.radians(5.0)
    l_corner_max_axis_residual_rad: float = math.radians(15.0)
    l_corner_image_crop_margin_px: int = 8
    l_corner_bev_crop_margin_m: float = 0.010

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, PalletGeometry):
            raise TypeError("geometry must be PalletGeometry")
        if not isinstance(self.gates, PalletPerceptionGates):
            raise TypeError("gates must be PalletPerceptionGates")
        for name in (
            "workspace_x_m",
            "workspace_y_m",
            "workspace_z_m",
            "stack_plane_z_m",
            "rim_outer_band_m",
        ):
            values = tuple(float(value) for value in getattr(self, name))
            if (
                len(values) != 2
                or not np.all(np.isfinite(values))
                or values[0] >= values[1]
            ):
                raise ValueError(f"{name} must be a finite increasing pair")
            object.__setattr__(self, name, values)
        for name in (
            "min_depth_m",
            "max_depth_m",
            "z_histogram_bin_m",
            "plane_seed_band_m",
            "plane_fit_tolerance_m",
            "plane_slab_m",
            "grid_resolution_m",
            "morphology_close_m",
            "morphology_dilate_m",
            "opening_component_min_m",
            "opening_component_max_m",
            "min_rim_support_ratio",
            "held_plane_min_separation_m",
            "held_plane_max_uncertainty_m",
            "l_corner_edge_band_m",
            "l_corner_min_front_support_m",
            "l_corner_min_side_support_m",
            "l_corner_max_line_p95_residual_m",
            "l_corner_max_connection_gap_m",
            "l_corner_max_orthogonality_error_rad",
            "l_corner_max_axis_residual_rad",
            "l_corner_bev_crop_margin_m",
        ):
            object.__setattr__(self, name, _finite_positive(getattr(self, name), name))
        if self.min_depth_m >= self.max_depth_m:
            raise ValueError("min_depth_m must be below max_depth_m")
        if self.opening_component_min_m >= self.opening_component_max_m:
            raise ValueError("opening component bounds must be increasing")
        if self.min_rim_support_ratio > 1.0:
            raise ValueError("min_rim_support_ratio must not exceed one")
        for name in ("plane_fit_max_points", "min_plane_points"):
            value = int(getattr(self, name))
            if value < 100:
                raise ValueError(f"{name} must be at least 100")
            object.__setattr__(self, name, value)
        crop_margin = int(self.l_corner_image_crop_margin_px)
        if crop_margin < 0:
            raise ValueError("l_corner_image_crop_margin_px must be non-negative")
        object.__setattr__(self, "l_corner_image_crop_margin_px", crop_margin)
        axis = np.asarray(self.rough_front_axis_base, dtype=np.float64)
        if (
            axis.shape != (2,)
            or not np.all(np.isfinite(axis))
            or np.linalg.norm(axis) <= 1e-9
        ):
            raise ValueError(
                "rough_front_axis_base must be a finite non-zero XY vector"
            )
        object.__setattr__(
            self, "rough_front_axis_base", tuple((axis / np.linalg.norm(axis)).tolist())
        )


@dataclass(frozen=True, slots=True)
class HeldBoxHint:
    """Optional measured EEF proxy used only to associate a closer top plane."""

    center_base: FloatArray | None = None
    yaw_base_rad: float | None = None
    eef_proxy_z_base_m: float | None = None
    footprint_size_m: tuple[float, float] = (0.400, 0.253)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "center_base", _vector(self.center_base, 3, "center_base")
        )
        if self.yaw_base_rad is not None and not math.isfinite(
            float(self.yaw_base_rad)
        ):
            raise ValueError("yaw_base_rad must be finite")
        if self.eef_proxy_z_base_m is not None and not math.isfinite(
            float(self.eef_proxy_z_base_m)
        ):
            raise ValueError("eef_proxy_z_base_m must be finite")
        object.__setattr__(
            self,
            "footprint_size_m",
            _float_pair(self.footprint_size_m, "footprint_size_m"),
        )


@dataclass(frozen=True, slots=True)
class BoundaryLineEvidence:
    """One observed metric boundary segment; it never implies a full rectangle."""

    role: str
    endpoints_base: FloatArray
    direction_base: FloatArray
    support_length_m: float
    support_point_count: int
    p95_residual_m: float
    axis_residual_rad: float
    touches_image_crop: bool
    touches_bev_crop: bool

    def __post_init__(self) -> None:
        role = str(self.role).strip()
        if not role:
            raise ValueError("boundary line role must be non-empty")
        object.__setattr__(self, "role", role)
        endpoints = _matrix(self.endpoints_base, 3, "endpoints_base")
        if endpoints is None or endpoints.shape != (2, 3):
            raise ValueError("endpoints_base must be a finite 2x3 matrix")
        object.__setattr__(self, "endpoints_base", endpoints)
        direction = _vector(self.direction_base, 3, "direction_base")
        if direction is None or float(np.linalg.norm(direction)) <= 1e-9:
            raise ValueError("direction_base must be non-zero")
        direction = direction / np.linalg.norm(direction)
        direction.setflags(write=False)
        object.__setattr__(self, "direction_base", direction)
        object.__setattr__(
            self,
            "support_length_m",
            _finite_positive(self.support_length_m, "support_length_m"),
        )
        count = int(self.support_point_count)
        if count < 2:
            raise ValueError("support_point_count must be at least two")
        object.__setattr__(self, "support_point_count", count)
        for name in ("p95_residual_m", "axis_residual_rad"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "touches_image_crop", bool(self.touches_image_crop))
        object.__setattr__(self, "touches_bev_crop", bool(self.touches_bev_crop))

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "endpoints_base_xyz_m": self.endpoints_base.tolist(),
            "direction_base": self.direction_base.tolist(),
            "support_length_m": self.support_length_m,
            "support_point_count": self.support_point_count,
            "p95_residual_m": self.p95_residual_m,
            "axis_residual_rad": self.axis_residual_rad,
            "axis_residual_deg": math.degrees(self.axis_residual_rad),
            "touches_image_crop": self.touches_image_crop,
            "touches_bev_crop": self.touches_bev_crop,
        }


@dataclass(frozen=True, slots=True)
class LCornerObservation:
    """Partial stack evidence with deliberately limited control authority.

    A valid observation constrains the selected near/right corner, stack-plane
    height and line orientation.  It intentionally has no stack-center, hole,
    or slot-target field, so it cannot be coerced into a fine-servo pose.
    """

    timestamp_s: float
    corner_base: FloatArray | None
    u_right_base: FloatArray | None
    v_far_base: FloatArray | None
    yaw_base_rad: float | None
    plane_height_base_m: float | None
    plane_p95_residual_m: float | None
    front_line: BoundaryLineEvidence | None
    side_line: BoundaryLineEvidence | None
    connection_gap_m: float | None
    orthogonality_error_rad: float | None
    topology_branch: str | None
    constrained_dofs: tuple[str, ...]
    unconstrained_dofs: tuple[str, ...]
    quality: Mapping[str, float]
    valid: bool
    rejection_reasons: tuple[str, ...]
    calibration_status: str = "nominal_ready_assumed"

    def __post_init__(self) -> None:
        timestamp = float(self.timestamp_s)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp_s must be finite")
        object.__setattr__(self, "timestamp_s", timestamp)
        for name in ("corner_base", "u_right_base", "v_far_base"):
            object.__setattr__(self, name, _vector(getattr(self, name), 3, name))
        for name in (
            "yaw_base_rad",
            "plane_height_base_m",
            "plane_p95_residual_m",
            "connection_gap_m",
            "orthogonality_error_rad",
        ):
            value = getattr(self, name)
            if value is not None and (
                not math.isfinite(float(value))
                or (name != "yaw_base_rad" and float(value) < 0.0)
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.front_line is not None and not isinstance(
            self.front_line, BoundaryLineEvidence
        ):
            raise TypeError("front_line must be BoundaryLineEvidence")
        if self.side_line is not None and not isinstance(
            self.side_line, BoundaryLineEvidence
        ):
            raise TypeError("side_line must be BoundaryLineEvidence")
        branch = (
            None if self.topology_branch is None else str(self.topology_branch).strip()
        )
        object.__setattr__(self, "topology_branch", branch or None)
        constrained = tuple(
            dict.fromkeys(str(value) for value in self.constrained_dofs if value)
        )
        unconstrained = tuple(
            dict.fromkeys(str(value) for value in self.unconstrained_dofs if value)
        )
        if not constrained or not unconstrained:
            raise ValueError(
                "L-corner observations require explicit constrained and unconstrained DOFs"
            )
        object.__setattr__(self, "constrained_dofs", constrained)
        object.__setattr__(self, "unconstrained_dofs", unconstrained)
        quality = {str(key): float(value) for key, value in self.quality.items()}
        if not all(math.isfinite(value) for value in quality.values()):
            raise ValueError("quality values must be finite")
        object.__setattr__(self, "quality", quality)
        reasons = tuple(
            dict.fromkeys(str(reason) for reason in self.rejection_reasons if reason)
        )
        object.__setattr__(self, "rejection_reasons", reasons)
        required = (
            self.corner_base,
            self.u_right_base,
            self.v_far_base,
            self.yaw_base_rad,
            self.plane_height_base_m,
            self.plane_p95_residual_m,
            self.front_line,
            self.side_line,
            self.connection_gap_m,
            self.orthogonality_error_rad,
            self.topology_branch,
        )
        if self.valid and (any(value is None for value in required) or reasons):
            raise ValueError(
                "valid L-corner observations require complete partial-line evidence"
            )
        if not self.valid and not reasons:
            raise ValueError("invalid L-corner observations require a rejection reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_s": self.timestamp_s,
            "corner_base_xyz_m": None
            if self.corner_base is None
            else self.corner_base.tolist(),
            "u_right_base": None
            if self.u_right_base is None
            else self.u_right_base.tolist(),
            "v_far_base": None if self.v_far_base is None else self.v_far_base.tolist(),
            "yaw_base_rad": self.yaw_base_rad,
            "yaw_base_deg": None
            if self.yaw_base_rad is None
            else math.degrees(self.yaw_base_rad),
            "plane_height_base_m": self.plane_height_base_m,
            "plane_p95_residual_m": self.plane_p95_residual_m,
            "front_line": None
            if self.front_line is None
            else self.front_line.to_dict(),
            "side_line": None if self.side_line is None else self.side_line.to_dict(),
            "connection_gap_m": self.connection_gap_m,
            "orthogonality_error_rad": self.orthogonality_error_rad,
            "orthogonality_error_deg": (
                None
                if self.orthogonality_error_rad is None
                else math.degrees(self.orthogonality_error_rad)
            ),
            "topology_branch": self.topology_branch,
            "constrained_dofs": list(self.constrained_dofs),
            "unconstrained_dofs": list(self.unconstrained_dofs),
            "quality": dict(self.quality),
            "valid": self.valid,
            "rejection_reasons": list(self.rejection_reasons),
            "calibration_status": self.calibration_status,
        }


@dataclass(frozen=True, slots=True)
class StackObservation:
    timestamp_s: float
    center_base: FloatArray | None
    u_right_base: FloatArray | None
    v_far_base: FloatArray | None
    yaw_base_rad: float | None
    plane_height_base_m: float | None
    slot1_target_base: FloatArray | None
    opening_size_m: tuple[float, float] | None
    quality: Mapping[str, float]
    valid: bool
    rejection_reasons: tuple[str, ...]
    calibration_status: str = "nominal_ready_assumed"
    axis_branch: str | None = None

    def __post_init__(self) -> None:
        timestamp = float(self.timestamp_s)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp_s must be finite")
        object.__setattr__(self, "timestamp_s", timestamp)
        for name in ("center_base", "u_right_base", "v_far_base", "slot1_target_base"):
            object.__setattr__(self, name, _vector(getattr(self, name), 3, name))
        for name in ("yaw_base_rad", "plane_height_base_m"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.opening_size_m is not None:
            object.__setattr__(
                self,
                "opening_size_m",
                _float_pair(self.opening_size_m, "opening_size_m"),
            )
        quality = {str(key): float(value) for key, value in self.quality.items()}
        if not all(math.isfinite(value) for value in quality.values()):
            raise ValueError("quality values must be finite")
        object.__setattr__(self, "quality", quality)
        reasons = tuple(
            dict.fromkeys(str(reason) for reason in self.rejection_reasons if reason)
        )
        object.__setattr__(self, "rejection_reasons", reasons)
        branch = None if self.axis_branch is None else str(self.axis_branch).strip()
        object.__setattr__(self, "axis_branch", branch or None)
        if self.valid and any(
            value is None
            for value in (
                self.center_base,
                self.u_right_base,
                self.v_far_base,
                self.yaw_base_rad,
                self.plane_height_base_m,
                self.slot1_target_base,
                self.opening_size_m,
                self.axis_branch,
            )
        ):
            raise ValueError(
                "valid stack observations require complete geometry and axis branch"
            )
        if not self.valid and not reasons:
            raise ValueError("invalid stack observations require a rejection reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_s": self.timestamp_s,
            "center_base_xyz_m": None
            if self.center_base is None
            else self.center_base.tolist(),
            "u_right_base": None
            if self.u_right_base is None
            else self.u_right_base.tolist(),
            "v_far_base": None if self.v_far_base is None else self.v_far_base.tolist(),
            "yaw_base_rad": self.yaw_base_rad,
            "yaw_base_deg": None
            if self.yaw_base_rad is None
            else math.degrees(self.yaw_base_rad),
            "plane_height_base_m": self.plane_height_base_m,
            "slot1_target_base_xyz_m": (
                None
                if self.slot1_target_base is None
                else self.slot1_target_base.tolist()
            ),
            "opening_size_m": None
            if self.opening_size_m is None
            else list(self.opening_size_m),
            "quality": dict(self.quality),
            "valid": self.valid,
            "rejection_reasons": list(self.rejection_reasons),
            "calibration_status": self.calibration_status,
            "axis_branch": self.axis_branch,
        }


@dataclass(frozen=True, slots=True)
class HeldBoxTopObservation:
    timestamp_s: float
    top_plane_z_base_m: float | None
    top_plane_z_uncertainty_m: float | None
    eef_proxy_z_base_m: float | None
    delta_z_top_eef_m: float | None
    footprint_size_m: tuple[float, float] | None
    distinct_from_stack: bool
    valid: bool
    rejection_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "timestamp_s",
            "top_plane_z_base_m",
            "top_plane_z_uncertainty_m",
            "eef_proxy_z_base_m",
            "delta_z_top_eef_m",
        ):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.footprint_size_m is not None:
            object.__setattr__(
                self,
                "footprint_size_m",
                _float_pair(self.footprint_size_m, "footprint_size_m"),
            )
        reasons = tuple(
            dict.fromkeys(str(reason) for reason in self.rejection_reasons if reason)
        )
        object.__setattr__(self, "rejection_reasons", reasons)
        if not self.valid and not reasons:
            raise ValueError("invalid held-top observations require a rejection reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_s": self.timestamp_s,
            "top_plane_z_base_m": self.top_plane_z_base_m,
            "top_plane_z_uncertainty_m": self.top_plane_z_uncertainty_m,
            "eef_proxy_z_base_m": self.eef_proxy_z_base_m,
            "delta_z_top_eef_m": self.delta_z_top_eef_m,
            "footprint_size_m": None
            if self.footprint_size_m is None
            else list(self.footprint_size_m),
            "distinct_from_stack": self.distinct_from_stack,
            "valid": self.valid,
            "rejection_reasons": list(self.rejection_reasons),
        }


@dataclass(frozen=True, slots=True)
class PalletSceneObservation:
    stack: StackObservation
    held_top: HeldBoxTopObservation | None
    coarse: LCornerObservation | None = None

    @property
    def valid(self) -> bool:
        return self.stack.valid

    def to_dict(self) -> dict[str, Any]:
        return {
            "stack": self.stack.to_dict(),
            "held_top": None if self.held_top is None else self.held_top.to_dict(),
            "coarse": None if self.coarse is None else self.coarse.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class SlotAlignmentObservation:
    held_center_base: FloatArray
    held_yaw_base_rad: float
    target_center_base: FloatArray
    target_yaw_base_rad: float
    raw_xy_error_m: FloatArray
    raw_yaw_error_rad: float

    def __post_init__(self) -> None:
        for name, length in (
            ("held_center_base", 3),
            ("target_center_base", 3),
            ("raw_xy_error_m", 2),
        ):
            object.__setattr__(self, name, _vector(getattr(self, name), length, name))
        for name in ("held_yaw_base_rad", "target_yaw_base_rad", "raw_yaw_error_rad"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class PalletFrameEvidence:
    """Bounded frame evidence retained for overlay and offline audit."""

    stack_plane_base: Plane | None = None
    opening_corners_base: FloatArray | None = None
    stack_points_base: FloatArray | None = None
    held_points_base: FloatArray | None = None
    rim_support_ratios: tuple[float, ...] = ()
    rim_observed: tuple[bool, ...] = ()
    outer_size_m: tuple[float, float] | None = None
    closer_points_rejected: int = 0
    held_excluded_points_base: FloatArray | None = None
    l_corner_component_points_base: FloatArray | None = None
    l_corner_front_endpoints_base: FloatArray | None = None
    l_corner_side_endpoints_base: FloatArray | None = None
    l_corner_corner_base: FloatArray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "opening_corners_base",
            _matrix(self.opening_corners_base, 3, "opening_corners_base"),
        )
        object.__setattr__(
            self,
            "stack_points_base",
            _matrix(self.stack_points_base, 3, "stack_points_base"),
        )
        object.__setattr__(
            self,
            "held_points_base",
            _matrix(self.held_points_base, 3, "held_points_base"),
        )
        object.__setattr__(
            self,
            "held_excluded_points_base",
            _matrix(self.held_excluded_points_base, 3, "held_excluded_points_base"),
        )
        object.__setattr__(
            self,
            "l_corner_component_points_base",
            _matrix(
                self.l_corner_component_points_base, 3, "l_corner_component_points_base"
            ),
        )
        for name in ("l_corner_front_endpoints_base", "l_corner_side_endpoints_base"):
            value = _matrix(getattr(self, name), 3, name)
            if value is not None and value.shape != (2, 3):
                raise ValueError(f"{name} must be a finite 2x3 matrix")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "l_corner_corner_base",
            _vector(self.l_corner_corner_base, 3, "l_corner_corner_base"),
        )
        ratios = tuple(float(value) for value in self.rim_support_ratios)
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in ratios):
            raise ValueError("rim_support_ratios must be finite fractions")
        object.__setattr__(self, "rim_support_ratios", ratios)
        object.__setattr__(
            self, "rim_observed", tuple(bool(value) for value in self.rim_observed)
        )
        if self.outer_size_m is not None:
            object.__setattr__(
                self, "outer_size_m", _float_pair(self.outer_size_m, "outer_size_m")
            )
        object.__setattr__(
            self, "closer_points_rejected", max(0, int(self.closer_points_rejected))
        )


# Replay fallback only.  Live code must replace this with fresh ready-pose FK.
NOMINAL_READY_T_BASE_FROM_HEAD = np.array(
    [
        [0.174400, 0.0, 0.984675, 0.583771],
        [0.0, 1.0, 0.0, 0.0],
        [-0.984675, 0.0, 0.174400, 1.130364],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
NOMINAL_READY_T_BASE_FROM_HEAD.setflags(write=False)


def load_pallet_estimator_config(
    root_config: Mapping[str, Any] | str | Path | PalletEstimatorConfig,
) -> PalletEstimatorConfig:
    """Load perception settings from the product root JSON contract.

    The checked-in product config uses top-level ``pallet`` and
    ``perception`` blocks.  A nested ``pallet.geometry`` form is accepted for
    callers that construct focused estimator configs in memory.
    """

    if isinstance(root_config, PalletEstimatorConfig):
        return root_config
    if isinstance(root_config, (str, Path)):
        path = Path(root_config)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read pallet configuration {path}: {exc}") from exc
        if not isinstance(value, Mapping):
            raise ValueError("pallet configuration root must be an object")
        root: Mapping[str, Any] = value
    elif isinstance(root_config, Mapping):
        root = root_config
    else:
        raise TypeError("root_config must be a mapping, path, or PalletEstimatorConfig")

    pallet_raw = root.get("pallet", {})
    if not isinstance(pallet_raw, Mapping):
        raise ValueError("pallet configuration block must be an object")
    geometry_raw = pallet_raw.get("geometry", pallet_raw)
    if not isinstance(geometry_raw, Mapping):
        raise ValueError("pallet geometry block must be an object")
    geometry = PalletGeometry(
        outer_size_m=tuple(geometry_raw.get("outer_size_m", (0.660, 0.658))),
        opening_size_m=tuple(geometry_raw.get("opening_size_m", (0.148, 0.149))),
        slot1_offset_m=tuple(
            geometry_raw.get(
                "slot1_offset_right_far_m",
                geometry_raw.get("slot1_offset_m", (0.12800, 0.20175)),
            )
        ),
    )

    perception_raw = root.get("perception", pallet_raw.get("perception", {}))
    if not isinstance(perception_raw, Mapping):
        raise ValueError("perception configuration block must be an object")
    gates = PalletPerceptionGates(
        min_inner_rim_count=int(perception_raw.get("minimum_inner_rims", 3)),
        max_opening_size_error_m=float(
            perception_raw.get("opening_dimension_tolerance_m", 0.015)
        ),
        max_orthogonality_error_rad=math.radians(
            float(perception_raw.get("maximum_orthogonality_error_deg", 5.0))
        ),
        max_plane_p95_residual_m=float(
            perception_raw.get("maximum_plane_p95_residual_m", 0.008)
        ),
        max_center_spread_m=float(perception_raw.get("live_center_spread_m", 0.008)),
        max_yaw_spread_rad=math.radians(
            float(perception_raw.get("live_yaw_spread_deg", 2.0))
        ),
        max_start_yaw_residual_rad=math.radians(
            float(
                perception_raw.get(
                    "maximum_initial_yaw_deg",
                    root.get("servo", {}).get("maximum_initial_yaw_deg", 15.0)
                    if isinstance(root.get("servo", {}), Mapping)
                    else 15.0,
                )
            )
        ),
        max_consecutive_center_jump_m=float(
            perception_raw.get(
                "maximum_measurement_jump_m",
                root.get("servo", {}).get("maximum_measurement_jump_m", 0.030)
                if isinstance(root.get("servo", {}), Mapping)
                else 0.030,
            )
        ),
    )

    defaults = PalletEstimatorConfig()
    kwargs: dict[str, Any] = {
        "geometry": geometry,
        "gates": gates,
        "min_depth_m": float(perception_raw.get("min_depth_m", defaults.min_depth_m)),
        "max_depth_m": float(perception_raw.get("max_depth_m", defaults.max_depth_m)),
        "plane_fit_tolerance_m": float(
            perception_raw.get(
                "plane_fit_tolerance_m",
                perception_raw.get(
                    "plane_ransac_tolerance_m", defaults.plane_fit_tolerance_m
                ),
            )
        ),
        "plane_slab_m": float(
            perception_raw.get("plane_slab_m", defaults.plane_slab_m)
        ),
        "min_plane_points": int(
            perception_raw.get("minimum_plane_points", defaults.min_plane_points)
        ),
        "held_plane_min_separation_m": float(
            perception_raw.get(
                "closer_plane_rejection_margin_m",
                defaults.held_plane_min_separation_m,
            )
        ),
    }
    for key in (
        "workspace_x_m",
        "workspace_y_m",
        "workspace_z_m",
        "stack_plane_z_m",
        "z_histogram_bin_m",
        "plane_seed_band_m",
        "plane_fit_max_points",
        "grid_resolution_m",
        "morphology_close_m",
        "morphology_dilate_m",
        "opening_component_min_m",
        "opening_component_max_m",
        "rim_outer_band_m",
        "min_rim_support_ratio",
        "held_plane_max_uncertainty_m",
        "rough_front_axis_base",
    ):
        if key in perception_raw:
            kwargs[key] = perception_raw[key]
    l_corner_raw = perception_raw.get("l_corner", root.get("l_corner", {}))
    acquisition_raw = root.get("acquisition", {})
    if (
        not l_corner_raw
        and isinstance(acquisition_raw, Mapping)
        and isinstance(acquisition_raw.get("l_corner", {}), Mapping)
    ):
        l_corner_raw = acquisition_raw.get("l_corner", {})
    if not isinstance(l_corner_raw, Mapping):
        raise ValueError("l_corner configuration block must be an object")
    for key in (
        "l_corner_edge_band_m",
        "l_corner_min_front_support_m",
        "l_corner_min_side_support_m",
        "l_corner_max_line_p95_residual_m",
        "l_corner_max_connection_gap_m",
        "l_corner_max_orthogonality_error_rad",
        "l_corner_max_axis_residual_rad",
        "l_corner_image_crop_margin_px",
        "l_corner_bev_crop_margin_m",
    ):
        short_key = key.removeprefix("l_corner_")
        if key in perception_raw:
            kwargs[key] = perception_raw[key]
        elif key in l_corner_raw:
            kwargs[key] = l_corner_raw[key]
        elif short_key in l_corner_raw:
            kwargs[key] = l_corner_raw[short_key]
    return PalletEstimatorConfig(**kwargs)


__all__ = [
    "BoundaryLineEvidence",
    "HeldBoxHint",
    "HeldBoxTopObservation",
    "LCornerObservation",
    "NOMINAL_READY_T_BASE_FROM_HEAD",
    "PalletEstimatorConfig",
    "PalletFrameEvidence",
    "PalletGeometry",
    "PalletPerceptionGates",
    "PalletSceneObservation",
    "SlotAlignmentObservation",
    "StackObservation",
    "load_pallet_estimator_config",
]
