"""Colour-agnostic box-top segmentation by height above the support plane.

Replaces the old yellow-HSV mask. Works for any box colour because it uses only
the depth-derived geometry of the scene.

Two stages make it robust to a slightly tilted support-plane fit:
  1. gate to "above the table" points (clearance .. box_height + margin) -- a
     wide, forgiving band that captures the whole box top even if the table
     normal is a little off;
  2. RANSAC the top face *itself* among those points. The top face is a large
     clean plane, so this recovers its true (horizontal) plane accurately; its
     inliers are exactly the top face rather than a thin height-contour strip.

Fallback ``top_plane_direct``: when almost nothing sits above the plane (the box
fills the frame and the RANSAC already locked onto the box top), the plane is
used directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .geometry import SupportPlane, even_subsample, fit_support_plane


@dataclass
class Segmentation:
    ok: bool
    selector: np.ndarray            # bool mask over the deprojected point arrays
    mask: np.ndarray | None         # image-space uint8 (0/255) of the box top
    top_plane: SupportPlane         # the top-face plane (basis for yaw)
    mode: str
    measured_height_m: float | None  # top-face height above the table
    reason: str | None = None
    info: dict[str, Any] = field(default_factory=dict)


def segment_box_top(
    points: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    plane: SupportPlane,
    image_shape: tuple[int, int],
    *,
    box_height_m: float = 0.150,
    clearance_m: float = 0.030,
    top_margin_m: float = 0.060,
    top_inlier_m: float = 0.012,
    band_m: float = 0.035,
    min_top_pixels: int = 400,
    morph_kernel: int = 5,
    rng: np.random.Generator | None = None,
) -> Segmentation:
    """Select the box-top pixels from an already-deprojected point set."""
    rng = rng or np.random.default_rng(0)
    height = plane.signed_height(points)

    lo, hi = clearance_m, box_height_m + top_margin_m
    above = (height > lo) & (height < hi)
    n_above = int(np.count_nonzero(above))

    if n_above >= min_top_pixels:
        cand = points[above]
        sub = even_subsample(cand.shape[0], 6000)
        top_plane = fit_support_plane(
            cand[sub], iterations=200, tolerance_m=0.006,
            min_inliers=max(min_top_pixels // 2, 50), rng=rng,
        )
        if top_plane is None:
            return _fail(above, plane, "top_plane_not_found", {"n_above": n_above})
        top_h = top_plane.signed_height(points)
        selector = above & (np.abs(top_h) <= top_inlier_m)
        mode = "table_relative"
        measured_height: float | None = float(plane.signed_height(top_plane.point.reshape(1, 3))[0])
    else:
        selector = np.abs(height) <= band_m
        top_plane = plane
        mode = "top_plane_direct"
        measured_height = None

    n_sel = int(np.count_nonzero(selector))
    if n_sel < min_top_pixels:
        return _fail(selector, top_plane, "too_few_top_points", {"n_above": n_above, "selected": n_sel}, mode)

    # Rasterise, close gaps, keep the largest connected component (single box;
    # multi-box just iterates the remaining components later).
    mask = np.zeros(image_shape, dtype=np.uint8)
    mask[rows[selector], cols[selector]] = 255
    if morph_kernel >= 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    ncomp, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if ncomp <= 1:
        return _fail(selector, top_plane, "no_connected_component", {"n_above": n_above}, mode)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep_label = int(np.argmax(areas)) + 1

    sel_idx = np.nonzero(selector)[0]
    keep = labels[rows[sel_idx], cols[sel_idx]] == keep_label
    final_selector = np.zeros_like(selector)
    final_selector[sel_idx[keep]] = True
    final_mask = np.where(labels == keep_label, 255, 0).astype(np.uint8)

    if int(np.count_nonzero(final_selector)) < min_top_pixels:
        return _fail(final_selector, top_plane, "component_too_small", {"n_above": n_above}, mode, final_mask)

    return Segmentation(
        True, final_selector, final_mask, top_plane, mode, measured_height, None,
        {
            "n_above": n_above,
            "component_area_px": int(areas[keep_label - 1]),
            "component_count": int(ncomp - 1),
            "top_points": int(np.count_nonzero(final_selector)),
        },
    )


def _fail(
    selector: np.ndarray,
    top_plane: SupportPlane,
    reason: str,
    info: dict[str, Any],
    mode: str = "table_relative",
    mask: np.ndarray | None = None,
) -> Segmentation:
    return Segmentation(False, selector, mask, top_plane, mode, None, reason, info)
