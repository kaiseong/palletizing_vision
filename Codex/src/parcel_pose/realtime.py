"""Low-latency D435 live view for the fixed RB-Y1 parcel task.

The estimator remains depth-native. Only the four fitted top edges are
projected into native RGB using the active RealSense factory extrinsics. The
default path is perception-only; an explicitly supplied automation consumer
may use current base poses without changing estimator or overlay semantics.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Protocol, TextIO

import numpy as np
from numpy.typing import NDArray

from .calibration import factory_extrinsics_to_transform
from .estimator import EstimationEvidence, ParcelPoseEstimator
from .evaluation import BasePoseDiagnostic, base_pose_from_estimate
from .models import Calibration, CameraIntrinsics, EstimatorConfig
from .projection import unproject_plane_points
from .realsense_adapter import D435StreamConfig, RealSenseAdapter
from .transforms import transform_points
from .visualization import project_points_to_pixels


ImageArray = NDArray[np.uint8]
FloatArray = NDArray[np.float64]
DEFAULT_WINDOW_NAME = "RB-Y1 Parcel Pose"


class LiveViewUnavailableError(RuntimeError):
    """Raised when the camera or OpenCV display path cannot run."""


class LivePoseAutomation(Protocol):
    """Opt-in consumer that owns robot motion and the final grasp handoff."""

    def start(self) -> None:
        """Connect and prepare without moving the mobile base."""

    def update(
        self,
        base_pose: BasePoseDiagnostic | None,
        *,
        pose_timestamp_s: float,
        now_s: float,
    ) -> bool:
        """Consume one timestamped pose and return whether handoff is ready."""

    def handoff(self) -> None:
        """Stop/release the base stream and execute the grasp exactly once."""

    def close(self) -> None:
        """Fail closed, release resources, and disconnect."""


def _cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise LiveViewUnavailableError(
            "OpenCV with HighGUI support is required for live-view"
        ) from exc
    return cv2


def _require_highgui(cv2: Any) -> None:
    required = ("namedWindow", "imshow", "waitKey", "destroyWindow")
    missing = [name for name in required if not callable(getattr(cv2, name, None))]
    if missing:
        raise LiveViewUnavailableError(
            "OpenCV HighGUI functions are unavailable: " + ", ".join(missing)
        )

    current_framework = getattr(cv2, "currentUIFramework", None)
    if callable(current_framework):
        try:
            if not str(current_framework()).strip():
                raise LiveViewUnavailableError(
                    "OpenCV has no GUI backend; remove opencv-python-headless and "
                    "use a GUI-capable JetPack/OpenCV build"
                )
        except LiveViewUnavailableError:
            raise
        except Exception:
            pass
    else:
        build_information = getattr(cv2, "getBuildInformation", None)
        if callable(build_information):
            gui_lines = [
                line.split(":", 1)[1].strip().upper()
                for line in str(build_information()).splitlines()
                if line.strip().startswith("GUI:")
            ]
            if gui_lines and gui_lines[0] in {"", "NONE", "NO"}:
                raise LiveViewUnavailableError(
                    "OpenCV has no GUI backend; remove opencv-python-headless and "
                    "use a GUI-capable JetPack/OpenCV build"
                )

    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        raise LiveViewUnavailableError(
            "no graphical display is available; set DISPLAY/WAYLAND_DISPLAY or run "
            "live-view from the Jetson desktop session"
        )


def live_overlay_lines(
    base_pose: BasePoseDiagnostic | None,
    estimator_latency_ms: float,
) -> tuple[str, str]:
    """Return the complete and deliberately minimal live text contract."""

    latency = float(estimator_latency_ms)
    if not np.isfinite(latency) or latency < 0.0:
        raise ValueError("estimator latency must be finite and non-negative")
    if base_pose is None:
        return (
            "x=-- m   y=-- m   z=-- m",
            f"yaw=-- deg   latency={latency:.1f} ms",
        )

    x_m, y_m, z_m = base_pose.box_center_xyz_m
    values = (x_m, y_m, z_m, base_pose.yaw_signed_deg)
    if not all(np.isfinite(float(value)) for value in values):
        raise ValueError("base pose values must be finite")
    return (
        f"x={x_m:+.3f} m   y={y_m:+.3f} m   z={z_m:+.3f} m",
        f"yaw={base_pose.yaw_signed_deg:+.1f} deg   latency={latency:.1f} ms",
    )


def _top_edge_polygon(
    evidence: EstimationEvidence | None,
    color_from_depth: FloatArray,
    color_intrinsics: CameraIntrinsics,
) -> NDArray[np.int32] | None:
    if evidence is None or evidence.rectangle.corners_xy_m is None:
        return None
    corners_depth = unproject_plane_points(
        evidence.rectangle.corners_xy_m,
        evidence.projection.plane,
        origin=evidence.projection.origin_3d_m,
        basis=(evidence.projection.basis_u_3d, evidence.projection.basis_v_3d),
    )
    corners_color = transform_points(corners_depth, color_from_depth)
    pixels = project_points_to_pixels(corners_color, color_intrinsics)
    if not np.isfinite(pixels).all():
        return None
    return np.rint(pixels).astype(np.int32).reshape(-1, 1, 2)


def _draw_live_text_panel(cv2: Any, image: ImageArray, lines: tuple[str, str]) -> None:
    panel_height = min(image.shape[0], 66)
    panel_width = min(image.shape[1], 570)
    panel = image[:panel_height, :panel_width].copy()
    panel[:] = (22, 22, 22)
    cv2.addWeighted(
        panel,
        0.78,
        image[:panel_height, :panel_width],
        0.22,
        0.0,
        image[:panel_height, :panel_width],
    )
    for index, line in enumerate(lines):
        y = 24 + 25 * index
        cv2.putText(
            image,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )


def _validate_camera_profile(calibration: Calibration, metadata: Any) -> None:
    profile = calibration.diagnostics.get("camera_profile")
    if not isinstance(profile, Mapping):
        return
    expected_serial = profile.get("serial")
    if expected_serial is not None and str(metadata.camera_serial) != str(
        expected_serial
    ):
        raise LiveViewUnavailableError(
            "calibration camera serial mismatch: expected "
            f"{expected_serial}, connected {metadata.camera_serial}"
        )


def draw_live_overlay(
    image_bgr: Any,
    base_pose: BasePoseDiagnostic | None,
    *,
    evidence: EstimationEvidence | None,
    color_from_depth: FloatArray,
    color_intrinsics: CameraIntrinsics,
    estimator_latency_ms: float,
    cv2_module: Any | None = None,
) -> ImageArray:
    """Draw only the fitted top edges and the five requested base-pose fields."""

    cv2 = _cv2() if cv2_module is None else cv2_module
    image = np.asarray(image_bgr)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("raw RGB image must be uint8 HxWx3 BGR")
    output = image.copy()
    polygon = _top_edge_polygon(evidence, color_from_depth, color_intrinsics)
    if polygon is not None:
        color = (40, 220, 255) if base_pose is not None else (0, 128, 255)
        cv2.polylines(output, [polygon], True, color, 2, cv2.LINE_AA)
    _draw_live_text_panel(
        cv2,
        output,
        live_overlay_lines(base_pose, estimator_latency_ms),
    )
    return output


def _open_video(path: Path | None, shape: tuple[int, int], fps: int) -> Any | None:
    if path is None:
        return None
    if path.exists():
        raise FileExistsError(f"refusing to overwrite live-view video: {path}")
    cv2 = _cv2()
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = shape
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )
    if not writer.isOpened():
        raise LiveViewUnavailableError(f"cannot open live-view MP4 writer: {path}")
    return writer


def _open_log(path: Path | None) -> TextIO | None:
    if path is None:
        return None
    if path.exists():
        raise FileExistsError(f"refusing to overwrite live-view telemetry: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("x", encoding="utf-8")


def _write_live_record(
    stream: TextIO | None,
    *,
    frame_index: int,
    depth_frame_number: int,
    depth_timestamp_ms: float,
    estimator_latency_ms: float,
    base_pose: BasePoseDiagnostic | None,
    handoff_ready: bool,
) -> None:
    if stream is None:
        return
    if base_pose is None:
        pose_record: dict[str, Any] | None = None
    else:
        x_m, y_m, z_m = base_pose.box_center_xyz_m
        pose_record = {
            "box_center_xyz_m": [float(x_m), float(y_m), float(z_m)],
            "yaw_signed_deg": float(base_pose.yaw_signed_deg),
        }
    stream.write(
        json.dumps(
            {
                "frame_index": int(frame_index),
                "depth_frame_number": int(depth_frame_number),
                "depth_timestamp_ms": float(depth_timestamp_ms),
                "estimator_latency_ms": float(estimator_latency_ms),
                "base_pose": pose_record,
                "handoff_ready": bool(handoff_ready),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    stream.flush()


def run_live_view(
    calibration: Calibration,
    estimator_config: EstimatorConfig,
    metadata_context: Mapping[str, Any],
    *,
    warmup_frames: int = 30,
    max_frames: int | None = None,
    fullscreen: bool = False,
    window_name: str = DEFAULT_WINDOW_NAME,
    automation: LivePoseAutomation | None = None,
    headless: bool = False,
    output_mp4: str | Path | None = None,
    log_jsonl: str | Path | None = None,
) -> int:
    """Run the D435/estimator/display loop and return the displayed frame count."""

    if calibration.T_base_from_depth is None:
        raise ValueError(
            "live-view requires a complete T_base_from_depth transform chain"
        )
    if max_frames is not None and max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if not window_name.strip():
        raise ValueError("--window-name cannot be empty")

    cv2 = _cv2()
    if not headless:
        _require_highgui(cv2)
    displayed_frames = 0
    window_created = False
    handoff_ready = False
    handoff_started = False
    user_cancelled = False
    video_writer: Any | None = None
    log_stream: TextIO | None = None
    stream_config = D435StreamConfig(
        align_color_to_depth=False,
        warmup_frames=warmup_frames,
    )
    try:
        log_stream = _open_log(None if log_jsonl is None else Path(log_jsonl))
        with RealSenseAdapter(stream_config) as camera:
            metadata = camera.session_metadata(**dict(metadata_context))
            _validate_camera_profile(calibration, metadata)
            if automation is not None:
                automation.start()
            estimator = ParcelPoseEstimator(
                metadata.depth_profile.intrinsics,
                calibration,
                estimator_config,
            )
            color_from_depth = factory_extrinsics_to_transform(metadata.depth_to_color)
            if not headless:
                try:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    window_created = True
                    if fullscreen:
                        cv2.setWindowProperty(
                            window_name,
                            cv2.WND_PROP_FULLSCREEN,
                            cv2.WINDOW_FULLSCREEN,
                        )
                except Exception as exc:
                    raise LiveViewUnavailableError(
                        f"OpenCV could not create the live-view window: {exc}"
                    ) from exc

            while max_frames is None or displayed_frames < max_frames:
                try:
                    frame = camera.capture()
                except RuntimeError as exc:
                    raise LiveViewUnavailableError(
                        f"D435 live capture failed: {exc}"
                    ) from exc
                pose_timestamp_s = time.monotonic()
                estimate_start = time.perf_counter()
                estimate = estimator.estimate(
                    frame.raw_depth_z16,
                    depth_scale=metadata.depth_scale_m,
                    timestamp_ms=frame.depth_timestamp_ms,
                    frame_id=frame.depth_frame_number,
                )
                estimator_ms = 1000.0 * (time.perf_counter() - estimate_start)
                base_pose = base_pose_from_estimate(estimate, calibration)
                if automation is not None:
                    now_s = time.monotonic()
                    handoff_ready = automation.update(
                        base_pose,
                        pose_timestamp_s=pose_timestamp_s,
                        now_s=now_s,
                    )
                overlay = draw_live_overlay(
                    frame.raw_color_bgr,
                    base_pose,
                    evidence=estimator.last_evidence,
                    color_from_depth=color_from_depth,
                    color_intrinsics=metadata.color_profile.intrinsics,
                    estimator_latency_ms=estimator_ms,
                    cv2_module=cv2,
                )
                if video_writer is None and output_mp4 is not None:
                    video_writer = _open_video(Path(output_mp4), overlay.shape[:2], 30)
                if video_writer is not None:
                    video_writer.write(overlay)
                _write_live_record(
                    log_stream,
                    frame_index=displayed_frames,
                    depth_frame_number=frame.depth_frame_number,
                    depth_timestamp_ms=frame.depth_timestamp_ms,
                    estimator_latency_ms=estimator_ms,
                    base_pose=base_pose,
                    handoff_ready=handoff_ready,
                )
                key = -1
                if not headless:
                    try:
                        cv2.imshow(window_name, overlay)
                        key = int(cv2.waitKey(1)) & 0xFF
                    except Exception as exc:
                        raise LiveViewUnavailableError(
                            f"OpenCV live-view display failed: {exc}"
                        ) from exc
                displayed_frames += 1
                if key in {27, ord("q"), ord("Q")}:
                    user_cancelled = True
                    handoff_ready = False
                    break
                if handoff_ready:
                    break
        if handoff_ready and not user_cancelled and automation is not None:
            handoff_started = True
            automation.handoff()
    except KeyboardInterrupt:
        user_cancelled = True
        if handoff_started:
            raise
    finally:
        if video_writer is not None:
            video_writer.release()
        if log_stream is not None:
            log_stream.close()
        if window_created:
            try:
                cv2.destroyWindow(window_name)
            except Exception:
                destroy_all = getattr(cv2, "destroyAllWindows", None)
                if callable(destroy_all):
                    destroy_all()
        if automation is not None:
            automation.close()
    return displayed_frames


__all__ = [
    "DEFAULT_WINDOW_NAME",
    "LivePoseAutomation",
    "LiveViewUnavailableError",
    "draw_live_overlay",
    "live_overlay_lines",
    "run_live_view",
]
