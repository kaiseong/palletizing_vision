"""Debug overlay for a BoxOrientation estimate."""

from __future__ import annotations

import numpy as np

from .geometry import CameraIntrinsics
from .orientation import BoxOrientation


def draw_overlay(bgr: np.ndarray, est: BoxOrientation, intr: CameraIntrinsics) -> np.ndarray:
    import cv2

    out = np.ascontiguousarray(bgr).copy()

    if est.ok and est.center_camera_m is not None:
        color = (0, 220, 0)
    elif est.center_camera_m is not None:
        color = (0, 165, 255)  # orange: geometry found but low confidence
    else:
        color = (0, 0, 255)

    if getattr(est, "top_mask", None) is not None:
        contours, _ = cv2.findContours((est.top_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (120, 120, 120), 1)

    if est.center_camera_m is not None and est.long_axis_camera is not None:
        c = np.asarray(est.center_camera_m, dtype=np.float64)
        long_v = np.asarray(est.long_axis_camera, dtype=np.float64) * (est.long_len_m or 0.0) * 0.5
        short_v = np.asarray(est.short_axis_camera, dtype=np.float64) * (est.short_len_m or 0.0) * 0.5
        corners3d = np.array([c - long_v - short_v, c + long_v - short_v, c + long_v + short_v, c - long_v + short_v])
        if np.all(corners3d[:, 2] > 0):
            pix = intr.project(corners3d).astype(np.int32)
            cv2.polylines(out, [pix.reshape(-1, 1, 2)], True, color, 2)
        if c[2] > 0 and (c + long_v)[2] > 0:
            p0 = intr.project(c[None])[0].astype(np.int32)
            p1 = intr.project((c + long_v)[None])[0].astype(np.int32)
            cv2.arrowedLine(out, tuple(p0), tuple(p1), (255, 80, 0), 2, tipLength=0.2)
            cv2.circle(out, tuple(p0), 4, color, -1)

    lines = _text_lines(est)
    y = 26
    for text in lines:
        cv2.putText(out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24
    return out


def _text_lines(est: BoxOrientation) -> list[str]:
    if est.reference is None or est.yaw_deg is None:
        return [f"FAIL: {', '.join(est.reasons) or 'no estimate'}"]
    lines = [
        f"ref={est.reference}deg  dev={est.deviation_deg:+.1f}  (yaw={est.yaw_deg:+.1f} {est.yaw_frame})",
        f"size LxS={est.long_len_m*1000:.0f}x{est.short_len_m*1000:.0f}mm  aspect={est.aspect:.2f}",
        f"conf={est.confidence:.2f}  pts={est.n_top_points}  mode={est.seg_mode}",
    ]
    if est.measured_height_m is not None:
        lines[1] += f"  h={est.measured_height_m*1000:.0f}mm"
    if est.reasons:
        lines.append("warn: " + ", ".join(est.reasons))
    return lines
