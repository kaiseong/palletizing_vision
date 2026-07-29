"""Perception-only overlays for metric pallet observations."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .models import CameraIntrinsics
from .pallet_acquisition import HoleGateStatus, LCornerGateStatus
from .pallet_models import (
    PalletFrameEvidence,
    PalletSceneObservation,
    Slot1HoleReference,
)
from .transforms import invert_transform, transform_points, validate_transform
from .visualization import project_points_to_pixels


ImageArray = NDArray[np.uint8]
FloatArray = NDArray[np.float64]


def _cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to render pallet overlays") from exc
    return cv2


def _as_bgr(image: ArrayLike) -> ImageArray:
    value = np.asarray(image)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("image must be uint8 HxWx3 BGR")
    return value.copy()


def _base_pixels(
    points_base: ArrayLike,
    intrinsics: CameraIntrinsics,
    T_base_depth: ArrayLike,
) -> FloatArray:
    depth_from_base = invert_transform(
        validate_transform(T_base_depth, name="T_base_depth")
    )
    points_depth = transform_points(points_base, depth_from_base)
    return project_points_to_pixels(points_depth, intrinsics)


def _inside_pixels(
    pixels: FloatArray, width: int, height: int
) -> tuple[FloatArray, NDArray[np.bool_]]:
    finite = np.isfinite(pixels).all(axis=1)
    rounded = np.zeros_like(pixels, dtype=np.int32)
    rounded[finite] = np.rint(pixels[finite]).astype(np.int32)
    inside = (
        finite
        & (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    return rounded, inside


def _draw_text_panel(image: ImageArray, lines: Sequence[str], *, warning: bool) -> None:
    cv2 = _cv2()
    panel_height = min(image.shape[0], 16 + 20 * len(lines))
    panel_width = min(image.shape[1], 620)
    panel = image[:panel_height, :panel_width].copy()
    panel[:] = (18, 18, 18)
    cv2.addWeighted(
        panel,
        0.78,
        image[:panel_height, :panel_width],
        0.22,
        0.0,
        image[:panel_height, :panel_width],
    )
    for index, line in enumerate(lines):
        color = (80, 190, 255) if warning and index == 1 else (245, 245, 245)
        y = 20 + index * 20
        cv2.putText(
            image,
            line,
            (9, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image, line, (9, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA
        )


def _field(value: object | None, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _draw_metric_segment(
    image: ImageArray,
    endpoints_base: ArrayLike | None,
    intrinsics: CameraIntrinsics,
    T_base_depth: ArrayLike,
    *,
    color: tuple[int, int, int],
    label: str,
) -> None:
    if endpoints_base is None:
        return
    cv2 = _cv2()
    endpoints = np.asarray(endpoints_base, dtype=np.float64)
    if endpoints.shape != (2, 3) or not np.all(np.isfinite(endpoints)):
        return
    pixels = _base_pixels(endpoints, intrinsics, T_base_depth)
    if not np.isfinite(pixels).all():
        return
    rounded = np.rint(pixels).astype(np.int32)
    cv2.line(
        image,
        tuple(rounded[0]),
        tuple(rounded[1]),
        color,
        4,
        cv2.LINE_AA,
    )
    midpoint = tuple(np.rint(np.mean(pixels, axis=0)).astype(np.int32))
    cv2.putText(
        image,
        label,
        (midpoint[0] + 6, midpoint[1] - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_pallet_overlay(
    image_bgr: ArrayLike,
    observation: PalletSceneObservation,
    *,
    evidence: PalletFrameEvidence | None,
    intrinsics: CameraIntrinsics,
    T_base_depth: ArrayLike,
    frame_id: int | None = None,
    state: str | None = None,
    latency_ms: float | None = None,
    l_corner_gate: LCornerGateStatus | Mapping[str, Any] | None = None,
    hole_gate: HoleGateStatus | Mapping[str, Any] | None = None,
    acquisition_audit: Mapping[str, Any] | None = None,
    slot1_hole_reference: Slot1HoleReference | None = None,
) -> ImageArray:
    """Draw selected stack plane, inner rims, axes, and slot-1 target.

    ``image_bgr`` must be color already aligned onto the depth pixel grid.
    This keeps the overlay on the exact raw-depth intrinsics used by geometry.
    """

    cv2 = _cv2()
    output = _as_bgr(image_bgr)
    height, width = output.shape[:2]

    if evidence is not None and evidence.stack_points_base is not None:
        pixels = _base_pixels(evidence.stack_points_base, intrinsics, T_base_depth)
        rounded, inside = _inside_pixels(pixels, width, height)
        output[rounded[inside, 1], rounded[inside, 0]] = (70, 170, 70)
    if evidence is not None and evidence.held_points_base is not None:
        pixels = _base_pixels(evidence.held_points_base, intrinsics, T_base_depth)
        rounded, inside = _inside_pixels(pixels, width, height)
        output[rounded[inside, 1], rounded[inside, 0]] = (115, 115, 115)

    if evidence is not None and evidence.l_corner_component_points_base is not None:
        pixels = _base_pixels(
            evidence.l_corner_component_points_base,
            intrinsics,
            T_base_depth,
        )
        rounded, inside = _inside_pixels(pixels, width, height)
        output[rounded[inside, 1], rounded[inside, 0]] = (180, 120, 45)
    if evidence is not None:
        _draw_metric_segment(
            output,
            evidence.l_corner_front_endpoints_base,
            intrinsics,
            T_base_depth,
            color=(255, 145, 20),
            label="L front",
        )
        _draw_metric_segment(
            output,
            evidence.l_corner_side_endpoints_base,
            intrinsics,
            T_base_depth,
            color=(40, 100, 255),
            label="L side",
        )
        if evidence.l_corner_corner_base is not None:
            corner_pixel = _base_pixels(
                [evidence.l_corner_corner_base],
                intrinsics,
                T_base_depth,
            )[0]
            if np.isfinite(corner_pixel).all():
                corner_uv = tuple(np.rint(corner_pixel).astype(int))
                if 0 <= corner_uv[0] < width and 0 <= corner_uv[1] < height:
                    cv2.drawMarker(
                        output,
                        corner_uv,
                        (70, 255, 70),
                        cv2.MARKER_DIAMOND,
                        20,
                        3,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        output,
                        "partial L corner",
                        (corner_uv[0] + 8, corner_uv[1] + 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.46,
                        (70, 255, 70),
                        2,
                        cv2.LINE_AA,
                    )

    if evidence is not None and evidence.opening_corners_base is not None:
        pixels = _base_pixels(evidence.opening_corners_base, intrinsics, T_base_depth)
        if np.isfinite(pixels).all():
            polygon = np.rint(pixels).astype(np.int32)
            for index in range(4):
                observed = (
                    index < len(evidence.rim_observed) and evidence.rim_observed[index]
                )
                color = (40, 230, 255) if observed else (0, 80, 255)
                cv2.line(
                    output,
                    tuple(polygon[index]),
                    tuple(polygon[(index + 1) % 4]),
                    color,
                    3,
                    cv2.LINE_AA,
                )

    stack = observation.stack
    if stack.center_base is not None:
        center_pixel = _base_pixels([stack.center_base], intrinsics, T_base_depth)[0]
        if np.isfinite(center_pixel).all():
            center_uv = tuple(np.rint(center_pixel).astype(int))
            if 0 <= center_uv[0] < width and 0 <= center_uv[1] < height:
                cv2.drawMarker(
                    output,
                    center_uv,
                    (40, 230, 255),
                    cv2.MARKER_CROSS,
                    16,
                    2,
                    cv2.LINE_AA,
                )
        if stack.u_right_base is not None and stack.v_far_base is not None:
            axis_points = np.stack(
                (
                    stack.center_base,
                    stack.center_base + 0.10 * stack.u_right_base,
                    stack.center_base + 0.10 * stack.v_far_base,
                )
            )
            axis_pixels = _base_pixels(axis_points, intrinsics, T_base_depth)
            if np.isfinite(axis_pixels).all():
                center_uv = tuple(np.rint(axis_pixels[0]).astype(int))
                cv2.arrowedLine(
                    output,
                    center_uv,
                    tuple(np.rint(axis_pixels[1]).astype(int)),
                    (255, 170, 20),
                    2,
                    cv2.LINE_AA,
                    tipLength=0.20,
                )
                cv2.arrowedLine(
                    output,
                    center_uv,
                    tuple(np.rint(axis_pixels[2]).astype(int)),
                    (20, 200, 255),
                    2,
                    cv2.LINE_AA,
                    tipLength=0.20,
                )
                cv2.putText(
                    output,
                    "u_right",
                    tuple(np.rint(axis_pixels[1]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 170, 20),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    output,
                    "v_far",
                    tuple(np.rint(axis_pixels[2]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (20, 200, 255),
                    1,
                    cv2.LINE_AA,
                )
    if stack.slot1_target_base is not None:
        target_pixel = _base_pixels(
            [stack.slot1_target_base], intrinsics, T_base_depth
        )[0]
        if np.isfinite(target_pixel).all():
            target_uv = tuple(np.rint(target_pixel).astype(int))
            if 0 <= target_uv[0] < width and 0 <= target_uv[1] < height:
                cv2.drawMarker(
                    output,
                    target_uv,
                    (255, 40, 210),
                    cv2.MARKER_TILTED_CROSS,
                    20,
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    output,
                    "geometric slot1",
                    (target_uv[0] + 8, target_uv[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    (255, 40, 210),
                    2,
                    cv2.LINE_AA,
                )

    if (
        slot1_hole_reference is not None
        and stack.center_base is not None
        and stack.plane_height_base_m is not None
    ):
        feature_points = np.asarray(
            (
                stack.center_base,
                (
                    slot1_hole_reference.center_base_xy_m[0],
                    slot1_hole_reference.center_base_xy_m[1],
                    stack.plane_height_base_m,
                ),
            ),
            dtype=np.float64,
        )
        feature_pixels = _base_pixels(feature_points, intrinsics, T_base_depth)
        rounded, inside = _inside_pixels(feature_pixels, width, height)
        if inside[0]:
            current_uv = tuple(int(value) for value in rounded[0])
            cv2.drawMarker(
                output,
                current_uv,
                (0, 255, 255),
                cv2.MARKER_TILTED_CROSS,
                18,
                2,
                cv2.LINE_AA,
            )
        if inside[1]:
            reference_uv = tuple(int(value) for value in rounded[1])
            cv2.drawMarker(
                output,
                reference_uv,
                (255, 40, 210),
                cv2.MARKER_CROSS,
                20,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                output,
                "demonstrated hole target",
                (reference_uv[0] + 8, reference_uv[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 40, 210),
                1,
                cv2.LINE_AA,
            )
        if inside[0] and inside[1]:
            cv2.line(
                output,
                tuple(int(value) for value in rounded[0]),
                tuple(int(value) for value in rounded[1]),
                (255, 40, 210),
                1,
                cv2.LINE_AA,
            )

    frame_text = "" if frame_id is None else f" frame={frame_id}"
    state_text = "" if state is None else f" state={state}"
    coarse_valid = bool(observation.coarse is not None and observation.coarse.valid)
    coarse = observation.coarse
    edge_pair_valid = bool(
        coarse is not None and coarse.forward_acquisition_valid
    )
    status = (
        "VALID"
        if stack.valid
        else "PARTIAL-L"
        if coarse_valid
        else "EDGE-PAIR"
        if edge_pair_valid
        else "ABSTAIN"
    )
    branch = stack.axis_branch or (None if coarse is None else coarse.topology_branch)
    source = getattr(stack, "stack_se2_source", None) or "--"
    lines = [
        f"PALLET SLOT1 {status}{frame_text}{state_text}",
        (
            f"registration: {stack.calibration_status}  "
            f"branch={branch or '--'} source={source}"
        ),
    ]
    if stack.center_base is not None and stack.slot1_target_base is not None:
        center = stack.center_base
        target = stack.slot1_target_base
        yaw_deg = (
            math.degrees(float(stack.yaw_base_rad))
            if stack.yaw_base_rad is not None
            else math.nan
        )
        lines.extend(
            (
                (
                    f"stack center [m] x={center[0]:+.3f} "
                    f"y={center[1]:+.3f} z={center[2]:+.3f} "
                    f"yaw={yaw_deg:+.2f} deg"
                ),
                (
                    f"slot1 base [m] x={target[0]:+.3f} "
                    f"y={target[1]:+.3f} z={target[2]:+.3f}"
                ),
            )
        )
        if slot1_hole_reference is not None:
            reference = slot1_hole_reference
            lines.append(
                "demonstrated hole ref [m] "
                f"x={reference.center_base_xy_m[0]:+.3f} "
                f"y={reference.center_base_xy_m[1]:+.3f} "
                f"yaw={math.degrees(reference.yaw_base_rad):+.2f} deg"
            )
    quality = stack.quality
    if quality and all(
        key in quality
        for key in (
            "opening_u_m",
            "opening_v_m",
            "stack_plane_p95_residual_m",
            "orthogonality_error_rad",
        )
    ):
        opening_u = quality.get("opening_u_m", math.nan)
        opening_v = quality.get("opening_v_m", math.nan)
        residual = 1_000.0 * quality.get("stack_plane_p95_residual_m", math.nan)
        orthogonality = math.degrees(quality.get("orthogonality_error_rad", math.nan))
        lines.append(
            f"opening={1_000*opening_u:.1f}x{1_000*opening_v:.1f} mm "
            f"rims={int(quality.get('inner_rim_count', 0))}/4 "
            f"plane_p95={residual:.1f} mm orth={orthogonality:.1f} deg"
        )
    if quality and "fixed_approach_signed_alignment" in quality:
        alignment = quality.get("fixed_approach_signed_alignment", math.nan)
        residual = math.degrees(
            quality.get("fixed_approach_axis_residual_rad", math.nan)
        )
        crosscheck = quality.get("opening_crosscheck_pass", math.nan)
        lines.append(
            f"fixed-axis align={alignment:+.3f} residual={residual:.1f} deg "
            f"opening_crosscheck={crosscheck:.0f}"
        )
    elif (coarse_valid or edge_pair_valid) and coarse is not None:
        front_support = (
            math.nan
            if coarse.front_line is None
            else coarse.front_line.support_length_m
        )
        side_support = (
            math.nan if coarse.side_line is None else coarse.side_line.support_length_m
        )
        residual = (
            math.nan
            if coarse.plane_p95_residual_m is None
            else 1_000.0 * coarse.plane_p95_residual_m
        )
        gap = (
            math.nan
            if coarse.connection_gap_m is None
            else 1_000.0 * coarse.connection_gap_m
        )
        orthogonality = (
            math.nan
            if coarse.orthogonality_error_rad is None
            else math.degrees(coarse.orthogonality_error_rad)
        )
        evidence_label = "partial L" if coarse_valid else "forward edge-pair"
        lines.append(
            f"{evidence_label} "
            f"front={front_support:.3f} m side={side_support:.3f} m "
            f"plane_p95={residual:.1f} mm"
        )
        lines.append(
            f"{evidence_label} connection_gap={gap:.1f} mm "
            f"orthogonality={orthogonality:.1f} deg"
        )
    if latency_ms is not None:
        lines.append(f"latency={float(latency_ms):.1f} ms  NO GT: repeatability only")
    else:
        lines.append("NO GT: repeatability only")
    if stack.rejection_reasons:
        lines.append("reason: " + ", ".join(stack.rejection_reasons[:3]))
    if acquisition_audit is not None:
        phase = str(acquisition_audit.get("phase", "--"))
        if bool(acquisition_audit.get("in_fixed_review_interval", False)):
            lines.append(
                "OFFLINE AUDIT stationary=ASSUMED "
                f"phase={phase} odometry=UNAVAILABLE"
            )
            lines.append(
                "L gate "
                f"{int(_field(l_corner_gate, 'stationary_frames', 0))}/5 "
                f"stable={bool(_field(l_corner_gate, 'stable', False))} "
                "metric_proxy="
                f"{bool(_field(l_corner_gate, 'metric_proxy_stable', False))} "
                "hole_dwell="
                f"{bool(_field(hole_gate, 'dwell_complete', False))}"
            )
            lines.append(
                "motion=BLOCKED cmd(vx,vy,wz)=(0,0,0) "
                "would_request proxy/forward="
                f"{bool(acquisition_audit.get('would_request_metric_proxy_handoff', False))}/"
                f"{bool(acquisition_audit.get('would_request_forward_step_from_geometry_only', False))}"
            )
        else:
            lines.append(
                "OFFLINE AUDIT outside L frames 72..105; "
                "hole_dwell="
                f"{bool(_field(hole_gate, 'dwell_complete', False))} motion=BLOCKED"
            )
    _draw_text_panel(
        output,
        lines,
        warning=(not stack.valid or stack.calibration_status != "validated"),
    )
    return output


__all__ = ["draw_pallet_overlay"]
