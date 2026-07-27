"""Box yaw estimation and the 0deg / 90deg reference convention.

Pipeline (colour-agnostic, metric):
  1. deproject valid depth in a ROI                       -> camera-frame points
  2. discover the support plane (desk / pallet)           -> normal toward camera
  3. segment the box top by height above that plane       -> top-face points
  4. project the top face onto the plane and minAreaRect  -> long / short axis
  5. express the long axis yaw about the plane normal in the reference frame
  6. wrap to [-45, 135) and classify 0-base vs 90-base + signed deviation

Because the box is 400x250 (aspect ~1.6) and symmetric, the long axis fixes the
orientation up to the 180deg symmetry, which the wrap resolves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import cv2
import numpy as np

from .geometry import (
    CameraIntrinsics,
    SupportPlane,
    deproject_valid,
    even_subsample,
    fit_support_plane,
    normalize,
    plane_basis,
    refine_plane,
)
from .segment import segment_box_top


@dataclass(frozen=True)
class OrientConfig:
    z_min: float = 0.20
    z_max: float = 1.50
    min_scene_points: int = 2000
    plane_sample: int = 5000
    ransac_iterations: int = 250
    ransac_tolerance_m: float = 0.006
    plane_min_inliers: int = 150
    plane_candidates: int = 4
    refine_plane: bool = True
    plane_refine_band_m: float = 0.010
    band_m: float = 0.035
    top_margin_m: float = 0.060
    top_inlier_m: float = 0.012
    min_top_pixels: int = 400
    # Optional expected support-plane normal in the camera frame (fixed camera).
    # RANSAC candidates whose normal deviates more than the tolerance are
    # rejected, which discards wall / floor planes in cluttered scenes.
    normal_hint: tuple[float, float, float] | None = None
    normal_hint_tol_deg: float = 30.0
    boundary_deg: float = 45.0
    min_aspect: float = 1.25
    size_tol_m: float = 0.05


@dataclass
class BoxOrientation:
    ok: bool
    reference: int | None = None            # 0 or 90
    deviation_deg: float | None = None      # signed deviation from the reference
    yaw_deg: float | None = None            # long-axis yaw wrapped to [-45, 135)
    yaw_raw_mod180: float | None = None
    yaw_frame: str = "camera"
    long_len_m: float | None = None
    short_len_m: float | None = None
    aspect: float | None = None
    measured_height_m: float | None = None
    seg_mode: str | None = None
    n_top_points: int = 0
    confidence: float = 0.0
    reasons: tuple[str, ...] = ()
    # geometry kept for visualisation / downstream (not serialised as arrays)
    center_camera_m: np.ndarray | None = None
    center_ref_m: np.ndarray | None = None
    long_axis_camera: np.ndarray | None = None
    short_axis_camera: np.ndarray | None = None
    plane: SupportPlane | None = None
    top_mask: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        def vec(v: np.ndarray | None) -> list[float] | None:
            return None if v is None else [float(x) for x in np.asarray(v).reshape(-1)]

        return {
            "ok": bool(self.ok),
            "reference": self.reference,
            "deviation_deg": _round(self.deviation_deg),
            "yaw_deg": _round(self.yaw_deg),
            "yaw_raw_mod180": _round(self.yaw_raw_mod180),
            "yaw_frame": self.yaw_frame,
            "long_len_m": _round(self.long_len_m, 4),
            "short_len_m": _round(self.short_len_m, 4),
            "aspect": _round(self.aspect, 3),
            "measured_height_m": _round(self.measured_height_m, 4),
            "seg_mode": self.seg_mode,
            "n_top_points": int(self.n_top_points),
            "confidence": _round(self.confidence, 3),
            "reasons": list(self.reasons),
            "center_camera_m": vec(self.center_camera_m),
            "center_ref_m": vec(self.center_ref_m),
            "long_axis_camera": vec(self.long_axis_camera),
        }


def estimate_box_orientation(
    depth_m: np.ndarray,
    intr: CameraIntrinsics,
    *,
    box_long_m: float = 0.400,
    box_short_m: float = 0.250,
    box_height_m: float = 0.150,
    camera_to_ref: np.ndarray | None = None,
    ref_zero: tuple[float, float, float] = (1.0, 0.0, 0.0),
    plane: SupportPlane | None = None,
    config: OrientConfig | None = None,
    seed: int = 12345,
) -> BoxOrientation:
    """Estimate the single-box yaw and its 0/90 reference from an RGB-D frame.

    ``camera_to_ref`` is a 4x4 camera->reference(T5) transform. When omitted the
    yaw is reported in the camera frame (still fine for relative servoing).
    ``ref_zero`` sets which reference-frame direction counts as 0deg.
    """
    cfg = config or OrientConfig()
    rng = np.random.default_rng(seed)

    points, rows, cols = deproject_valid(depth_m, intr, z_min=cfg.z_min, z_max=cfg.z_max)
    if points.shape[0] < cfg.min_scene_points:
        return _fail("insufficient_depth", {"scene_points": int(points.shape[0])})

    sub = even_subsample(points.shape[0], cfg.plane_sample)
    used_plane = plane
    if used_plane is None:
        used_plane = discover_support_plane(points[sub], box_height_m, cfg, rng)
        if used_plane is None:
            return _fail("no_support_plane", {})
    if cfg.refine_plane:
        used_plane = refine_plane(used_plane, points[sub], band_m=cfg.plane_refine_band_m)

    seg = segment_box_top(
        points, rows, cols, used_plane, depth_m.shape,
        box_height_m=box_height_m,
        top_margin_m=cfg.top_margin_m,
        top_inlier_m=cfg.top_inlier_m,
        band_m=cfg.band_m,
        min_top_pixels=cfg.min_top_pixels,
        rng=rng,
    )
    if not seg.ok:
        return _fail(seg.reason or "segmentation_failed", seg.info, plane=used_plane, seg_mode=seg.mode)

    top_plane = seg.top_plane
    top_pts = points[seg.selector]
    e1, e2 = plane_basis(top_plane.normal)
    origin = top_plane.point
    coords = np.column_stack(((top_pts - origin) @ e1, (top_pts - origin) @ e2)).astype(np.float32)

    rect = cv2.minAreaRect(coords)
    box2d = cv2.boxPoints(rect)
    center2d = np.asarray(rect[0], dtype=np.float64)
    long2d, short2d, long_len, short_len = _axes_from_box2d(box2d)
    aspect = float(long_len / max(short_len, 1e-9))

    long3d = normalize(long2d[0] * e1 + long2d[1] * e2)
    short3d = normalize(short2d[0] * e1 + short2d[1] * e2)
    center3d = origin + center2d[0] * e1 + center2d[1] * e2

    yaw_raw, yaw_frame = yaw_about_normal(long3d, top_plane.normal, camera_to_ref, ref_zero)
    reference, deviation, yaw_wrapped = classify_0_90(yaw_raw, cfg.boundary_deg)

    center_ref = None
    if camera_to_ref is not None:
        center_ref = (np.asarray(camera_to_ref, dtype=np.float64) @ np.append(center3d, 1.0))[:3]

    reasons: list[str] = []
    if aspect < cfg.min_aspect:
        reasons.append("aspect_ambiguous")
    size_err = max(abs(long_len - box_long_m), abs(short_len - box_short_m))
    if size_err > cfg.size_tol_m:
        reasons.append("size_mismatch")
    if seg.mode == "top_plane_direct":
        reasons.append("table_not_observed")

    confidence = _confidence(reasons, aspect, size_err, top_pts.shape[0], used_plane)
    return BoxOrientation(
        ok=not reasons,
        reference=reference,
        deviation_deg=deviation,
        yaw_deg=yaw_wrapped,
        yaw_raw_mod180=yaw_raw,
        yaw_frame=yaw_frame,
        long_len_m=float(long_len),
        short_len_m=float(short_len),
        aspect=aspect,
        measured_height_m=seg.measured_height_m,
        seg_mode=seg.mode,
        n_top_points=int(top_pts.shape[0]),
        confidence=confidence,
        reasons=tuple(reasons),
        center_camera_m=center3d,
        center_ref_m=center_ref,
        long_axis_camera=long3d,
        short_axis_camera=short3d,
        plane=top_plane,
        top_mask=seg.mask,
    )


# --------------------------------------------------------------------------- #
# Support-plane discovery (multi-candidate, robust to walls/floor)
# --------------------------------------------------------------------------- #
def discover_support_plane(
    points: np.ndarray,
    box_height_m: float,
    cfg: OrientConfig,
    rng: np.random.Generator,
) -> SupportPlane | None:
    """Pick the plane the box rests on, not just the biggest plane.

    RANSAC several planes (peeling inliers between rounds) and prefer the one
    with the most points sitting a box-height above it -- that is the surface
    the box stands on. Falls back to the largest plane when nothing shows a
    box-height cluster (box fills the frame; top_plane_direct handles it later).
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    remaining = np.ones(pts.shape[0], dtype=bool)
    best: SupportPlane | None = None
    best_score = -1.0
    # Reward planes that carry a box-height cluster above them -- that is the
    # surface the box stands on (the table), not a wall or the box top itself.
    lo, hi = box_height_m * 0.6, box_height_m * 1.5
    hint = None if cfg.normal_hint is None else normalize(np.asarray(cfg.normal_hint, dtype=np.float64))
    cos_tol = math.cos(math.radians(cfg.normal_hint_tol_deg))

    for _ in range(cfg.plane_candidates):
        idx = np.nonzero(remaining)[0]
        if idx.size < cfg.plane_min_inliers:
            break
        candidate = fit_support_plane(
            pts[idx],
            iterations=cfg.ransac_iterations,
            tolerance_m=cfg.ransac_tolerance_m,
            min_inliers=cfg.plane_min_inliers,
            rng=rng,
        )
        if candidate is None:
            break
        height = candidate.signed_height(pts)
        inliers = np.abs(height) <= cfg.ransac_tolerance_m
        remaining &= ~inliers
        if hint is not None and abs(float(candidate.normal @ hint)) < cos_tol:
            continue  # wall / floor: wrong orientation for a support surface
        above_box = int(np.count_nonzero((height > lo) & (height < hi)))
        score = float(above_box) + 0.05 * float(np.count_nonzero(inliers))
        if score > best_score:
            best_score = score
            best = candidate
    return best


# --------------------------------------------------------------------------- #
# Yaw + convention
# --------------------------------------------------------------------------- #
def yaw_about_normal(
    long_axis_cam: np.ndarray,
    normal_cam: np.ndarray,
    camera_to_ref: np.ndarray | None,
    ref_zero: tuple[float, float, float],
) -> tuple[float, str]:
    """Yaw of the long axis about the plane normal, mod 180deg.

    Measured in the reference frame when ``camera_to_ref`` is given (rotation
    only), else in the camera frame. ``ref_zero`` is the 0deg direction.
    """
    if camera_to_ref is not None:
        R = np.asarray(camera_to_ref, dtype=np.float64)[:3, :3]
        long_r = R @ long_axis_cam
        n_r = R @ normal_cam
        frame = "ref"
    else:
        long_r, n_r, frame = np.asarray(long_axis_cam, float), np.asarray(normal_cam, float), "camera"

    n_r = normalize(n_r)
    zero = np.asarray(ref_zero, dtype=np.float64)
    zero_ip = zero - float(zero @ n_r) * n_r
    if float(np.linalg.norm(zero_ip)) < 1e-6:
        alt = np.array([0.0, 0.0, 1.0]) if abs(float(n_r[2])) < 0.9 else np.array([1.0, 0.0, 0.0])
        zero_ip = alt - float(alt @ n_r) * n_r
    zero_ip = normalize(zero_ip)
    ninety_ip = normalize(np.cross(n_r, zero_ip))

    long_ip = long_r - float(long_r @ n_r) * n_r
    if float(np.linalg.norm(long_ip)) < 1e-9:
        return math.nan, frame
    long_ip = normalize(long_ip)
    yaw = math.degrees(math.atan2(float(long_ip @ ninety_ip), float(long_ip @ zero_ip))) % 180.0
    return float(yaw), frame


def classify_0_90(yaw_mod180: float, boundary_deg: float = 45.0) -> tuple[int, float, float]:
    """Map a mod-180 yaw to (reference 0|90, signed deviation, wrapped yaw).

    Wrapped yaw lies in [-boundary, 180-boundary) = [-45, 135) by default:
      * [-45, 45)  -> reference 0,  deviation = yaw
      * [ 45, 135) -> reference 90, deviation = yaw - 90
    """
    if yaw_mod180 is None or (isinstance(yaw_mod180, float) and math.isnan(yaw_mod180)):
        return 0, math.nan, math.nan
    y = float(yaw_mod180) % 180.0
    if y >= 90.0 + boundary_deg:
        y -= 180.0
    if -boundary_deg <= y < boundary_deg:
        return 0, y, y
    return 90, y - 90.0, y


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _axes_from_box2d(box2d: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    pts = np.asarray(box2d, dtype=np.float64).reshape(4, 2)
    edges = [pts[(i + 1) % 4] - pts[i] for i in range(4)]
    lengths = [float(np.linalg.norm(e)) for e in edges]
    li = int(np.argmax(lengths))
    long_dir = normalize(edges[li])
    short_dir = np.array([-long_dir[1], long_dir[0]], dtype=np.float64)
    long_len = lengths[li]
    short_len = min(lengths)
    return long_dir, short_dir, long_len, short_len


def _confidence(
    reasons: list[str],
    aspect: float,
    size_err_m: float,
    n_top_points: int,
    plane: SupportPlane,
) -> float:
    score = 0.30
    score += 0.25 * min(max((aspect - 1.0) / 0.6, 0.0), 1.0)
    score += 0.25 * math.exp(-((size_err_m / 0.03) ** 2))
    score += 0.20 * min(max(n_top_points / 2000.0, 0.0), 1.0)
    if reasons:
        score *= 0.5
    return float(round(min(max(score, 0.0), 1.0), 3))


def _round(value: float | None, ndigits: int = 2) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return round(float(value), ndigits)


def _fail(
    reason: str,
    info: dict[str, Any],
    *,
    plane: SupportPlane | None = None,
    seg_mode: str | None = None,
) -> BoxOrientation:
    return BoxOrientation(
        ok=False,
        reasons=(reason,),
        seg_mode=seg_mode,
        plane=plane,
        confidence=0.0,
    )
