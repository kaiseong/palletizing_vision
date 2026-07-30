"""Frame-level D435 parcel center/yaw estimator.

Raw depth is the canonical geometric evidence.  Depth selects a calibrated
top-plane slab; the selected raw-depth pixel rays are then intersected with
the exact plane before the fixed-size rectangle is fitted.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from parcel_pose_common.angles import classify_canonical_angle_deg
from parcel_pose_common.models import (
    Calibration,
    CameraIntrinsics,
    EstimatorConfig,
    Plane,
    PoseEstimate,
)
from parcel_pose_common.plane import offset_plane
from .projection import DepthPlaneProjector, PlaneProjection, unproject_plane_points
from .rectangle_fit import RectangleFitConfig, RectangleFitResult, fit_fixed_rectangle
from parcel_pose_common.transforms import transform_directions, transform_points


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class EstimationEvidence:
    projection: PlaneProjection
    rectangle: RectangleFitResult
    selected_component: int | None
    component_sizes: tuple[int, ...]


def _rectangle_config(config: EstimatorConfig) -> RectangleFitConfig:
    # EstimatorConfig.outlier_quantile is the retained central fraction.  The
    # rectangle primitive expresses the same setting as a tail quantile.
    tail = max(0.001, min(0.03, 0.5 * (1.0 - float(config.outlier_quantile))))
    return RectangleFitConfig(
        min_points=int(config.min_points),
        max_fit_points=int(config.max_points),
        coarse_angle_step_deg=float(config.coarse_angle_step_deg),
        fine_angle_step_deg=float(config.fine_angle_step_deg),
        fine_half_width_deg=max(2.0, float(config.coarse_angle_step_deg) * 1.1),
        containment_tolerance_m=float(config.rectangle_containment_tolerance_m),
        edge_band_m=float(config.edge_band_m),
        robust_quantile=tail,
        border_margin_px=int(config.border_margin_px),
        full_extent_tolerance_m=max(0.020, 1.5 * float(config.edge_band_m)),
        min_side_span_ratio=float(config.min_side_span),
        min_axis_assignment_margin=float(config.min_candidate_margin),
    )


def _plane_in_depth(calibration: Calibration, plane: Plane) -> Plane:
    """Express a calibrated depth/base plane in the raw-depth optical frame."""

    depth_names = {"depth", calibration.depth_frame}
    base_names = {"base", calibration.base_frame}
    if plane.frame in depth_names:
        return Plane(normal=plane.normal, d=plane.d, frame=calibration.depth_frame)
    if plane.frame not in base_names:
        raise ValueError(
            f"table plane frame {plane.frame!r} is neither depth nor base frame"
        )
    transform = calibration.T_base_from_depth
    if transform is None:
        raise ValueError("base-frame table plane requires a complete T_base_from_depth chain")
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    normal_base = np.asarray(plane.normal, dtype=np.float64)
    normal_depth = rotation.T @ normal_base
    d_depth = float(plane.d) - float(normal_base @ translation)
    return Plane(normal=normal_depth, d=d_depth, frame=calibration.depth_frame)


def _component_selection(
    projection: PlaneProjection,
    *,
    min_points: int,
) -> tuple[NDArray[np.bool_], int | None, tuple[int, ...], tuple[str, ...]]:
    """Keep the dominant connected top-slab component in raw image space."""

    count = projection.count
    if count == 0:
        return np.zeros(0, dtype=np.bool_), None, (), ()
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        return np.zeros(count, dtype=np.bool_), None, (), ("component_filter_unavailable",)

    number, labels, stats, _ = cv2.connectedComponentsWithStats(
        projection.mask.astype(np.uint8),
        connectivity=8,
    )
    candidates: list[tuple[int, int]] = []
    for label in range(1, int(number)):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= int(min_points):
            candidates.append((area, label))
    if not candidates:
        return np.zeros(count, dtype=np.bool_), None, (), ()
    candidates.sort(key=lambda item: (-item[0], item[1]))
    selected_label = candidates[0][1]
    pixels = np.rint(projection.pixels_uv).astype(np.int64)
    selected = labels[pixels[:, 1], pixels[:, 0]] == selected_label
    reasons: list[str] = []
    if len(candidates) > 1 and candidates[1][0] >= 0.55 * candidates[0][0]:
        reasons.append("multiple_or_ambiguous_components")
    return selected, selected_label, tuple(area for area, _ in candidates), tuple(reasons)


def _workspace_keep(
    points_depth: FloatArray,
    calibration: Calibration,
    workspace: tuple[float, float, float, float] | None,
) -> tuple[NDArray[np.bool_], str | None]:
    if workspace is None:
        return np.ones(points_depth.shape[0], dtype=np.bool_), None
    transform = calibration.T_base_from_depth
    if transform is None:
        return np.ones(points_depth.shape[0], dtype=np.bool_), "workspace_ignored_without_base_transform"
    points_base = transform_points(points_depth, transform)
    xmin, xmax, ymin, ymax = workspace
    keep = (
        (points_base[:, 0] >= xmin)
        & (points_base[:, 0] <= xmax)
        & (points_base[:, 1] >= ymin)
        & (points_base[:, 1] <= ymax)
    )
    return keep, None


def _failure(
    reason: str,
    *,
    calibration: Calibration,
    config: EstimatorConfig,
    timestamp_ms: float,
    frame_id: int,
    diagnostics: dict[str, Any] | None = None,
) -> PoseEstimate:
    base_registration = (
        "validated"
        if calibration.absolute_base_validated
        else "nominal_unverified"
        if calibration.has_base_transform_chain
        else "unavailable"
    )
    merged_diagnostics = _dimension_diagnostics(config)
    if diagnostics is not None:
        merged_diagnostics.update(diagnostics)
    return PoseEstimate(
        timestamp_ms=timestamp_ms,
        frame_id=frame_id,
        frame=calibration.depth_frame,
        box_model=config.box_model,
        observability={
            "center_long": "underconstrained",
            "center_short": "underconstrained",
            "yaw": "unavailable",
            "reference": "unavailable",
        },
        per_field_confidence={"center_long": 0.0, "center_short": 0.0, "yaw": 0.0},
        diagnostics=merged_diagnostics,
        reasons=(reason,),
        calibration_state=calibration.state,
        base_registration=base_registration,
        base_registration_valid=calibration.absolute_base_validated,
    )


def _dimension_diagnostics(config: EstimatorConfig) -> dict[str, Any]:
    """Expose the physical source behind the active fixed-size model."""

    model = config.box_model
    prior = config.box_dimension_prior
    result: dict[str, Any] = {
        "box_dimensions": {
            "model": model.to_dict(),
            "inference": (
                "fixed_population_representative"
                if prior is not None
                else "fixed_configured_model"
            ),
        }
    }
    if prior is not None:
        prior_summary = prior.to_dict()
        prior_summary.pop("samples")
        result["box_dimensions"]["prior"] = prior_summary
        result["box_dimensions"]["per_frame_size_adaptation"] = False
        result["box_dimensions"]["height_adaptation_requires"] = (
            "explicit_on_table_arms_clear_lifecycle"
        )
    return result


class ParcelPoseEstimator:
    """Hardware-independent, deterministic frame estimator."""

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        calibration: Calibration,
        config: EstimatorConfig | None = None,
    ) -> None:
        self.intrinsics = intrinsics
        self.calibration = calibration
        self.config = EstimatorConfig() if config is None else config
        self.last_evidence: EstimationEvidence | None = None
        self._rectangle_fit_config = _rectangle_config(self.config)
        self._table_depth: Plane | None = None
        self._top_depth: Plane | None = None
        self._top_projector: DepthPlaneProjector | None = None
        self._static_geometry_failure: tuple[str, str] | None = None
        if self.calibration.table_plane is not None:
            try:
                self._table_depth = _plane_in_depth(
                    self.calibration,
                    self.calibration.table_plane,
                )
            except ValueError as exc:
                self._static_geometry_failure = (
                    "table_plane_frame_unresolved",
                    str(exc),
                )
            else:
                self._top_depth = offset_plane(
                    self._table_depth,
                    self.config.box_model.height_m,
                )
                try:
                    self._top_projector = DepthPlaneProjector(
                        self.intrinsics,
                        self._top_depth,
                    )
                except ValueError as exc:
                    self._static_geometry_failure = (
                        "invalid_depth_or_metadata",
                        str(exc),
                    )

    def estimate(
        self,
        depth: ArrayLike,
        *,
        depth_scale: float | None = None,
        timestamp_ms: float = 0.0,
        frame_id: int = 0,
        rgb_support_mask: ArrayLike | None = None,
    ) -> PoseEstimate:
        """Estimate one pose from raw Z16 or metric raw-depth data."""

        # Evidence belongs to exactly one input frame.  Clearing it before any
        # early-return path prevents a live viewer from drawing an old parcel
        # rectangle over a newer failed/empty frame.
        self.last_evidence = None

        if self.calibration.table_plane is None:
            return _failure(
                "table_plane_missing",
                calibration=self.calibration,
                config=self.config,
                timestamp_ms=timestamp_ms,
                frame_id=frame_id,
            )
        if self._static_geometry_failure is not None:
            reason, detail = self._static_geometry_failure
            return _failure(
                reason,
                calibration=self.calibration,
                config=self.config,
                timestamp_ms=timestamp_ms,
                frame_id=frame_id,
                diagnostics={"detail": detail},
            )
        if (
            self._table_depth is None
            or self._top_depth is None
            or self._top_projector is None
        ):
            return _failure(
                "invalid_depth_or_metadata",
                calibration=self.calibration,
                config=self.config,
                timestamp_ms=timestamp_ms,
                frame_id=frame_id,
                diagnostics={"detail": "fixed top-plane projector is unavailable"},
            )
        try:
            projection = self._top_projector.project(
                depth,
                depth_scale=depth_scale,
                slab_tolerance_m=self.config.top_plane_tolerance_m,
                min_depth_m=self.config.min_depth_m,
                max_depth_m=self.config.max_depth_m,
                support_mask=rgb_support_mask,
                max_points=self.config.max_points,
            )
        except ValueError as exc:
            return _failure(
                "invalid_depth_or_metadata",
                calibration=self.calibration,
                config=self.config,
                timestamp_ms=timestamp_ms,
                frame_id=frame_id,
                diagnostics={"detail": str(exc)},
            )
        if projection.count < self.config.min_points:
            return _failure(
                "insufficient_top_plane_points",
                calibration=self.calibration,
                config=self.config,
                timestamp_ms=timestamp_ms,
                frame_id=frame_id,
                diagnostics=projection.diagnostics,
            )

        component_keep, component_label, component_sizes, component_reasons = _component_selection(
            projection,
            min_points=self.config.min_points,
        )
        if "component_filter_unavailable" in component_reasons:
            self.last_evidence = None
            return _failure(
                "component_filter_unavailable",
                calibration=self.calibration,
                config=self.config,
                timestamp_ms=timestamp_ms,
                frame_id=frame_id,
                diagnostics={
                    "projection": projection.diagnostics,
                    "component_sizes": list(component_sizes),
                    "selected_component": component_label,
                },
            )
        workspace_keep, workspace_reason = _workspace_keep(
            projection.points_3d_m,
            self.calibration,
            self.config.workspace_xy_m,
        )
        keep = component_keep & workspace_keep
        points_xy = projection.points_xy_m[keep]
        pixels = projection.pixels_uv[keep]
        rectangle = fit_fixed_rectangle(
            points_xy,
            self.config.box_model,
            pixels_uv=pixels,
            image_shape=(self.intrinsics.height, self.intrinsics.width),
            config=self._rectangle_fit_config,
        )
        self.last_evidence = EstimationEvidence(
            projection=projection,
            rectangle=rectangle,
            selected_component=component_label,
            component_sizes=component_sizes,
        )

        reasons = list(rectangle.reasons)
        reasons.extend(component_reasons)
        if workspace_reason is not None:
            reasons.append(workspace_reason)

        center_depth: tuple[float, float, float] | None = None
        center_base: tuple[float, float] | None = None
        top_center_base: tuple[float, float, float] | None = None
        box_center_base: tuple[float, float, float] | None = None
        long_base: tuple[float, float] | None = None
        short_base: tuple[float, float] | None = None
        nominal_base: dict[str, Any] | None = None
        yaw_output = rectangle.yaw_rad
        output_frame = "table_plane"
        long_3d: FloatArray | None = None
        short_3d: FloatArray | None = None
        center_3d: FloatArray | None = None
        if rectangle.yaw_rad is not None:
            long_plane = np.asarray(rectangle.long_axis_xy, dtype=np.float64)
            short_plane = np.asarray(rectangle.short_axis_xy, dtype=np.float64)
            long_3d = long_plane[0] * projection.basis_u_3d + long_plane[1] * projection.basis_v_3d
            short_3d = short_plane[0] * projection.basis_u_3d + short_plane[1] * projection.basis_v_3d
        if rectangle.center_xy_m is not None:
            center_3d = unproject_plane_points(
                [rectangle.center_xy_m],
                self._top_depth,
                origin=projection.origin_3d_m,
                basis=(projection.basis_u_3d, projection.basis_v_3d),
            )[0]
            center_depth = tuple(float(value) for value in center_3d)

        base_transform = self.calibration.T_base_from_depth
        if base_transform is not None and long_3d is not None and short_3d is not None:
            transformed_axes = transform_directions(np.stack((long_3d, short_3d)), base_transform)
            long_xy_raw = transformed_axes[0, :2]
            short_xy_raw = transformed_axes[1, :2]
            long_xy_raw /= max(float(np.linalg.norm(long_xy_raw)), 1e-12)
            short_xy_raw /= max(float(np.linalg.norm(short_xy_raw)), 1e-12)
            yaw_base = float(math.atan2(long_xy_raw[1], long_xy_raw[0]) % math.pi)
            nominal_base = {
                "yaw_rad": yaw_base,
                "long_axis_xy": long_xy_raw.tolist(),
                "short_axis_xy": short_xy_raw.tolist(),
            }
            if center_3d is not None:
                center_base_3d = transform_points(center_3d, base_transform)
                table_normal_base = base_transform[:3, :3] @ self._table_depth.normal
                table_normal_base /= max(float(np.linalg.norm(table_normal_base)), 1e-12)
                box_center_base_3d = (
                    center_base_3d
                    - 0.5 * float(self.config.box_model.height_m) * table_normal_base
                )
                nominal_base["center_xy_m"] = center_base_3d[:2].tolist()
                nominal_base["center_xyz_m"] = center_base_3d.tolist()
                nominal_base["top_center_xyz_m"] = center_base_3d.tolist()
                nominal_base["box_center_xyz_m"] = box_center_base_3d.tolist()
            if self.calibration.absolute_base_validated:
                yaw_output = yaw_base
                long_base = tuple(float(value) for value in long_xy_raw)
                short_base = tuple(float(value) for value in short_xy_raw)
                if center_3d is not None:
                    center_base = tuple(float(value) for value in center_base_3d[:2])
                    top_center_base = tuple(float(value) for value in center_base_3d)
                    box_center_base = tuple(float(value) for value in box_center_base_3d)
                output_frame = self.calibration.base_frame

        yaw_degrees = None if yaw_output is None else math.degrees(yaw_output) % 180.0
        yaw_uncertainty = 0.5 * float(self.config.fine_angle_step_deg)
        canonical = classify_canonical_angle_deg(
            yaw_degrees,
            uncertainty_deg=yaw_uncertainty,
            long_short_assignment_valid=rectangle.valid_yaw,
        )
        observability = dict(rectangle.observability)
        observability["reference"] = canonical.status
        if canonical.status != "constrained":
            reasons.append(canonical.status)

        base_registration = (
            "validated"
            if self.calibration.absolute_base_validated
            else "nominal_unverified"
            if self.calibration.has_base_transform_chain
            else "unavailable"
        )
        if not self.calibration.absolute_base_validated:
            reasons.append("absolute_base_transform_unvalidated")

        component_ambiguous = "multiple_or_ambiguous_components" in reasons
        fit_confidence = (
            0.0
            if component_ambiguous
            else float(np.clip(1.0 - rectangle.score, 0.0, 1.0))
        )
        long_conf = fit_confidence if rectangle.observability["center_long"] != "underconstrained" else 0.0
        short_conf = fit_confidence if rectangle.observability["center_short"] != "underconstrained" else 0.0
        yaw_conf = (
            float(np.clip(rectangle.candidate_margin / 0.25, 0.0, 1.0)) * fit_confidence
            if rectangle.valid_yaw
            else 0.0
        )
        feasible_set = rectangle.feasible_set
        diagnostics: dict[str, Any] = {
            **_dimension_diagnostics(self.config),
            "projection": projection.diagnostics,
            "rectangle": rectangle.to_dict(),
            "component_sizes": list(component_sizes),
            "selected_component": component_label,
            "yaw_frame": output_frame,
        }
        if nominal_base is not None and not self.calibration.absolute_base_validated:
            diagnostics["nominal_unverified_base"] = nominal_base

        geometry_valid = rectangle.geometry_valid and not component_ambiguous
        full_pose_valid = rectangle.full_pose_valid and not component_ambiguous
        base_valid = self.calibration.absolute_base_validated and base_transform is not None
        return PoseEstimate(
            timestamp_ms=timestamp_ms,
            frame_id=frame_id,
            frame=output_frame,
            box_model=self.config.box_model,
            center_plane_xy_m=rectangle.center_xy_m,
            center_depth_m=center_depth,
            center_base_xy_m=center_base,
            top_center_base_xyz_m=top_center_base,
            box_center_base_xyz_m=box_center_base,
            yaw_rad=yaw_output,
            yaw_mod_180_deg=yaw_degrees,
            canonical_reference_deg=canonical.reference_deg,
            canonical_residual_deg=canonical.residual_deg,
            classification_margin_deg=canonical.classification_margin_deg,
            long_axis_plane_xy=rectangle.long_axis_xy,
            short_axis_plane_xy=rectangle.short_axis_xy,
            long_axis_base_xy=long_base,
            short_axis_base_xy=short_base,
            observability=observability,
            feasible_set=feasible_set,
            per_field_confidence={
                "center_long": long_conf,
                "center_short": short_conf,
                "yaw": yaw_conf,
                "reference": yaw_conf if canonical.status == "constrained" else 0.0,
            },
            diagnostics=diagnostics,
            reasons=tuple(dict.fromkeys(reasons)),
            calibration_state=self.calibration.state,
            base_registration=base_registration,
            geometry_valid=geometry_valid,
            full_pose_valid=full_pose_valid,
            base_registration_valid=base_valid,
            absolute_valid=full_pose_valid and base_valid,
        )


__all__ = ["EstimationEvidence", "ParcelPoseEstimator"]
