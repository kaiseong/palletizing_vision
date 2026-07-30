"""Stationary fresh-frame aggregation for parcel pose estimates."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np

from parcel_pose_common.angles import classify_canonical_angle_deg, line_angle_difference_rad
from parcel_pose_common.models import CalibrationState, PoseEstimate


@dataclass(frozen=True)
class BurstConfig:
    min_valid_frames: int = 3
    max_center_jitter_m: float = 0.020
    max_yaw_jitter_deg: float = 4.0

    def __post_init__(self) -> None:
        if self.min_valid_frames < 1:
            raise ValueError("min_valid_frames must be positive")
        if self.max_center_jitter_m <= 0.0 or self.max_yaw_jitter_deg <= 0.0:
            raise ValueError("burst jitter thresholds must be positive")


def _line_mean(angles: Sequence[float]) -> float:
    values = np.asarray(angles, dtype=np.float64)
    sine = float(np.mean(np.sin(2.0 * values)))
    cosine = float(np.mean(np.cos(2.0 * values)))
    if math.hypot(sine, cosine) <= 1e-12:
        raise ValueError("line angles have no identifiable circular mean")
    return float((0.5 * math.atan2(sine, cosine)) % math.pi)


def _tuple_median(values: Sequence[Sequence[float]], length: int) -> tuple[float, ...] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != length:
        return None
    return tuple(float(value) for value in np.median(array, axis=0))


def _center_jitter(values: Sequence[Sequence[float]], center: Sequence[float] | None) -> float:
    if not values or center is None:
        return 0.0
    array = np.asarray(values, dtype=np.float64)
    residual = np.linalg.norm(array - np.asarray(center, dtype=np.float64), axis=1)
    return float(np.max(residual))


def _state_rank(state: CalibrationState) -> int:
    return {
        CalibrationState.NOMINAL: 0,
        CalibrationState.PLANE_CALIBRATED_PARTIAL: 1,
        CalibrationState.BASE_VALIDATED: 2,
    }[state]


def aggregate_pose_burst(
    estimates: Iterable[PoseEstimate],
    *,
    min_valid_frames: int = 3,
    max_center_jitter_m: float = 0.020,
    max_yaw_jitter_deg: float = 4.0,
    min_timestamp_ms: float | None = None,
) -> PoseEstimate:
    """Aggregate a stationary fresh burst without manufacturing missing DOFs.

    Invalid/stale frames do not vote for a field.  A field is emitted only
    when it has ``min_valid_frames`` fresh observations and its temporal
    jitter is inside the configured limit.
    """

    settings = BurstConfig(
        min_valid_frames=min_valid_frames,
        max_center_jitter_m=max_center_jitter_m,
        max_yaw_jitter_deg=max_yaw_jitter_deg,
    )
    all_estimates = list(estimates)
    if not all_estimates:
        return PoseEstimate(
            observability={
                "center_long": "unavailable",
                "center_short": "unavailable",
                "yaw": "unavailable",
                "reference": "unavailable",
            },
            reasons=("empty_burst",),
            diagnostics={"burst": {"input_frames": 0, "fresh_frames": 0}},
        )

    # Preserve deterministic ordering and discard duplicate frame identities.
    ordered = sorted(all_estimates, key=lambda item: (item.timestamp_ms, item.frame_id))
    unique: list[PoseEstimate] = []
    seen: set[tuple[float, int]] = set()
    for estimate in ordered:
        identity = (estimate.timestamp_ms, estimate.frame_id)
        if identity in seen:
            continue
        seen.add(identity)
        if min_timestamp_ms is not None and estimate.timestamp_ms < float(min_timestamp_ms):
            continue
        if "stale_frame" in estimate.reasons or estimate.diagnostics.get("fresh") is False:
            continue
        unique.append(estimate)

    template = unique[-1] if unique else ordered[-1]
    if not unique:
        return PoseEstimate(
            timestamp_ms=template.timestamp_ms,
            frame_id=template.frame_id,
            frame=template.frame,
            box_model=template.box_model,
            calibration_state=template.calibration_state,
            base_registration=template.base_registration,
            observability={
                "center_long": "unavailable",
                "center_short": "unavailable",
                "yaw": "unavailable",
                "reference": "unavailable",
            },
            reasons=("no_fresh_frames",),
            diagnostics={
                "burst": {"input_frames": len(all_estimates), "fresh_frames": 0}
            },
        )

    frames = {item.frame for item in unique}
    reasons: list[str] = []
    if len(frames) != 1:
        reasons.append("mixed_output_frames")

    yaw_values = [float(item.yaw_rad) for item in unique if item.yaw_rad is not None]
    yaw_mean: float | None = None
    yaw_jitter_deg = 0.0
    if len(frames) == 1 and len(yaw_values) >= settings.min_valid_frames:
        try:
            yaw_mean = _line_mean(yaw_values)
            yaw_jitter_deg = max(
                abs(math.degrees(line_angle_difference_rad(value, yaw_mean)))
                for value in yaw_values
            )
        except ValueError:
            yaw_mean = None
            reasons.append("yaw_burst_ambiguous")
        if yaw_mean is not None and yaw_jitter_deg > settings.max_yaw_jitter_deg:
            yaw_mean = None
            reasons.append("temporal_jitter_too_large")
            reasons.append("yaw_temporal_jitter_too_large")
    else:
        reasons.append("insufficient_valid_yaw_frames")

    center_plane_values = [item.center_plane_xy_m for item in unique if item.center_plane_xy_m is not None]
    center_depth_values = [item.center_depth_m for item in unique if item.center_depth_m is not None]
    center_base_values = [item.center_base_xy_m for item in unique if item.center_base_xy_m is not None]
    top_center_base_values = [
        item.top_center_base_xyz_m
        for item in unique
        if item.top_center_base_xyz_m is not None
    ]
    box_center_base_values = [
        item.box_center_base_xyz_m
        for item in unique
        if item.box_center_base_xyz_m is not None
    ]
    center_plane = (
        _tuple_median(center_plane_values, 2)
        if len(center_plane_values) >= settings.min_valid_frames
        else None
    )
    center_depth = (
        _tuple_median(center_depth_values, 3)
        if len(center_depth_values) >= settings.min_valid_frames
        else None
    )
    center_base = (
        _tuple_median(center_base_values, 2)
        if len(center_base_values) >= settings.min_valid_frames
        else None
    )
    top_center_base = (
        _tuple_median(top_center_base_values, 3)
        if len(top_center_base_values) >= settings.min_valid_frames
        else None
    )
    box_center_base = (
        _tuple_median(box_center_base_values, 3)
        if len(box_center_base_values) >= settings.min_valid_frames
        else None
    )
    center_values_for_jitter: Sequence[Sequence[float]]
    center_for_jitter: Sequence[float] | None
    if center_base is not None:
        center_values_for_jitter, center_for_jitter = center_base_values, center_base
    else:
        center_values_for_jitter, center_for_jitter = center_plane_values, center_plane
    center_jitter = _center_jitter(center_values_for_jitter, center_for_jitter)
    if center_for_jitter is not None and center_jitter > settings.max_center_jitter_m:
        center_plane = None
        center_depth = None
        center_base = None
        top_center_base = None
        box_center_base = None
        reasons.append("temporal_jitter_too_large")
        reasons.append("center_temporal_jitter_too_large")
    elif center_for_jitter is None:
        reasons.append("insufficient_valid_center_frames")

    # A line direction has sign symmetry.  Reconstruct axes from the aggregate
    # yaw in the frame where yaw is declared, while separately aggregating
    # plane directions for plane-space consumers.
    long_plane_angles = [
        math.atan2(item.long_axis_plane_xy[1], item.long_axis_plane_xy[0])
        for item in unique
        if item.long_axis_plane_xy is not None
    ]
    long_plane: tuple[float, float] | None = None
    short_plane: tuple[float, float] | None = None
    if len(long_plane_angles) >= settings.min_valid_frames:
        angle = _line_mean(long_plane_angles)
        long_plane = (math.cos(angle), math.sin(angle))
        short_plane = (-math.sin(angle), math.cos(angle))
    long_base: tuple[float, float] | None = None
    short_base: tuple[float, float] | None = None
    if yaw_mean is not None and center_base is not None:
        long_base = (math.cos(yaw_mean), math.sin(yaw_mean))
        short_base = (-math.sin(yaw_mean), math.cos(yaw_mean))

    yaw_deg = None if yaw_mean is None else math.degrees(yaw_mean) % 180.0
    canonical = classify_canonical_angle_deg(
        yaw_deg,
        uncertainty_deg=yaw_jitter_deg,
        long_short_assignment_valid=yaw_mean is not None,
    )
    if canonical.status != "constrained":
        reasons.append(canonical.status)

    calibration_state = min(
        (item.calibration_state for item in unique),
        key=_state_rank,
    )
    all_base_registration_valid = all(item.base_registration_valid for item in unique)
    full_pose = yaw_mean is not None and (center_plane is not None or center_base is not None)
    absolute = (
        full_pose
        and all_base_registration_valid
        and calibration_state is CalibrationState.BASE_VALIDATED
        and center_base is not None
    )

    confidences: dict[str, float] = {}
    for name in ("center_long", "center_short", "yaw", "reference"):
        values = [float(item.per_field_confidence[name]) for item in unique if name in item.per_field_confidence]
        confidences[name] = float(np.median(values)) if values else 0.0
    if yaw_mean is None:
        confidences["yaw"] = confidences["reference"] = 0.0
    if center_plane is None and center_base is None:
        confidences["center_long"] = confidences["center_short"] = 0.0

    observations = {
        "center_long": "both_edges" if center_for_jitter is not None else "underconstrained",
        "center_short": "both_edges" if center_for_jitter is not None else "underconstrained",
        "yaw": "constrained" if yaw_mean is not None else "underconstrained",
        "reference": canonical.status,
    }
    # Aggregation can reduce temporal noise, but it cannot turn a physically
    # inferred crop edge into two observed edges.  Preserve the most
    # conservative provenance among frames that contributed a center.
    for axis in ("center_long", "center_short"):
        states = [
            item.observability.get(axis, "unavailable")
            for item in unique
            if item.center_plane_xy_m is not None or item.center_base_xy_m is not None
        ]
        if not states or any(
            state in ("underconstrained", "unavailable") for state in states
        ):
            observations[axis] = "underconstrained"
        elif any(state == "one_edge_inferred" for state in states):
            observations[axis] = "one_edge_inferred"

    inherited_reasons = [reason for item in unique for reason in item.reasons]
    reasons.extend(inherited_reasons)
    diagnostics = {
        "burst": {
            "input_frames": len(all_estimates),
            "fresh_frames": len(unique),
            "valid_yaw_frames": len(yaw_values),
            "valid_center_frames": len(center_values_for_jitter),
            "yaw_jitter_deg": yaw_jitter_deg,
            "center_jitter_m": center_jitter,
            "min_valid_frames": settings.min_valid_frames,
        }
    }
    return PoseEstimate(
        timestamp_ms=unique[-1].timestamp_ms,
        frame_id=unique[-1].frame_id,
        frame=unique[-1].frame,
        box_model=unique[-1].box_model,
        center_plane_xy_m=center_plane,
        center_depth_m=center_depth,
        center_base_xy_m=center_base,
        top_center_base_xyz_m=top_center_base,
        box_center_base_xyz_m=box_center_base,
        yaw_rad=yaw_mean,
        yaw_mod_180_deg=yaw_deg,
        canonical_reference_deg=canonical.reference_deg,
        canonical_residual_deg=canonical.residual_deg,
        classification_margin_deg=canonical.classification_margin_deg,
        long_axis_plane_xy=long_plane,
        short_axis_plane_xy=short_plane,
        long_axis_base_xy=long_base,
        short_axis_base_xy=short_base,
        observability=observations,
        feasible_set=None if full_pose else unique[-1].feasible_set,
        per_field_confidence=confidences,
        diagnostics=diagnostics,
        reasons=tuple(dict.fromkeys(reasons)),
        calibration_state=calibration_state,
        base_registration=unique[-1].base_registration,
        geometry_valid=yaw_mean is not None or center_for_jitter is not None,
        full_pose_valid=full_pose,
        base_registration_valid=all_base_registration_valid,
        absolute_valid=absolute,
    )


__all__ = ["BurstConfig", "aggregate_pose_burst"]
