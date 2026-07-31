"""Low-latency D435 live view for the fixed RB-Y1 parcel task.

The estimator remains depth-native. Only the four fitted top edges are
projected into native RGB using the active RealSense factory extrinsics. The
default path is perception-only; an explicitly supplied automation consumer
may use current base poses without changing estimator or overlay semantics.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any, Iterator, Mapping, Protocol, TextIO

import numpy as np
from numpy.typing import NDArray

from parcel_pose_common.calibration import factory_extrinsics_to_transform
from parcel_pose_common.models import Calibration, CameraIntrinsics, PoseResult
from parcel_pose_common.realsense_adapter import D435StreamConfig, RealSenseAdapter
from parcel_pose_common.transforms import transform_points
from parcel_pose_common.visualization import project_points_to_pixels

from .box_perception import perceive_box_pose
from .estimator import EstimationEvidence, ParcelPoseEstimator
from .evaluation import BasePoseDiagnostic
from .projection import unproject_plane_points


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


def _base_diagnostic_from_pose_result(
    pose_result: PoseResult,
) -> BasePoseDiagnostic | None:
    """Bridge the narrow facade result to the unchanged motion consumer.

    Motion decisions take x, y, and yaw from ``PoseResult``. The richer
    diagnostic retained by the facade supplies display-only z/top/registration
    fields so existing overlays and telemetry keep their current shape.
    """

    if not pose_result.valid:
        return None

    payload = pose_result.diagnostics.get("base_pose")
    if not isinstance(payload, Mapping):
        raise LiveViewUnavailableError(
            "valid box PoseResult is missing base_pose diagnostics"
        )
    try:
        box_center = payload["box_center_xyz_m"]
        top_center = payload["top_center_xyz_m"]
        assert pose_result.x_m is not None
        assert pose_result.y_m is not None
        assert pose_result.yaw_rad is not None
        return BasePoseDiagnostic(
            box_center_xyz_m=(
                float(pose_result.x_m),
                float(pose_result.y_m),
                float(box_center[2]),
            ),
            top_center_xyz_m=(
                float(top_center[0]),
                float(top_center[1]),
                float(top_center[2]),
            ),
            yaw_mod_180_deg=math.degrees(float(pose_result.yaw_rad)) % 180.0,
            yaw_signed_deg=float(payload["long_axis_yaw_base_signed_deg"]),
            canonical_reference_deg=(
                None
                if payload.get("canonical_reference_deg") is None
                else int(payload["canonical_reference_deg"])
            ),
            canonical_residual_deg=(
                None
                if payload.get("canonical_residual_deg") is None
                else float(payload["canonical_residual_deg"])
            ),
            registration=str(payload["registration"]),
        )
    except (AssertionError, IndexError, KeyError, TypeError, ValueError) as exc:
        raise LiveViewUnavailableError(
            f"invalid base_pose diagnostics in box PoseResult: {exc}"
        ) from exc


def _cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise LiveViewUnavailableError(
            "OpenCV is required to render box-picking overlays; install OpenCV or "
            "run --headless without --output-mp4"
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
            "box_picking.py from the Jetson desktop session, or pass --headless"
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


def _prepare_output_path(path: Path | None, *, description: str) -> Path | None:
    if path is None:
        return None
    prepared = path.expanduser()
    if prepared.exists():
        raise FileExistsError(f"refusing to overwrite {description}: {prepared}")
    prepared.parent.mkdir(parents=True, exist_ok=True)
    return prepared


def _open_video(
    path: Path | None,
    shape: tuple[int, int],
    fps: int,
    *,
    cv2_module: Any,
) -> Any | None:
    if path is None:
        return None
    height, width = shape
    writer = cv2_module.VideoWriter(
        str(path),
        cv2_module.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        writer.release()
        raise LiveViewUnavailableError(f"cannot open box-picking MP4 writer: {path}")
    return writer


def _open_log(path: Path | None) -> TextIO | None:
    if path is None:
        return None
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
            allow_nan=False,
        )
        + "\n"
    )
    stream.flush()


@dataclass(frozen=True)
class LiveViewPlan:
    """What a validated live-view request resolved to."""

    cv2: Any
    handoff_ready: Any
    handoff_started: Any
    log_jsonl_path: Any
    log_stream: Any
    needs_overlay: Any
    output_mp4_path: Any
    processed_frames: Any
    stream_config: Any
    user_cancelled: Any
    video_writer: Any
    window_created: Any


def resolve_live_view_plan(
    *,
    calibration: Any,
    fullscreen: Any,
    headless: Any,
    log_jsonl: Any,
    max_frames: Any,
    output_mp4: Any,
    warmup_frames: Any,
    window_name: Any,
) -> LiveViewPlan:
    """Check the request and the calibration, then hand back what the run needs.

    Pure: every refusal here happens before the camera opens, so a missing base
    transform or an unusable output path costs nothing.
    """

    cv2 = None
    handoff_ready = None
    handoff_started = None
    log_jsonl_path = None
    log_stream = None
    needs_overlay = None
    output_mp4_path = None
    processed_frames = None
    stream_config = None
    user_cancelled = None
    video_writer = None
    window_created = None

    if calibration.T_base_from_depth is None:
        raise ValueError(
            "box_picking.py requires a complete T_base_from_depth transform chain"
        )
    if max_frames is not None and max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if not window_name.strip():
        raise ValueError("--window-name cannot be empty")
    if headless and fullscreen:
        raise ValueError("--fullscreen cannot be combined with --headless")

    output_mp4_path = _prepare_output_path(
        None if output_mp4 is None else Path(output_mp4),
        description="box-picking video",
    )
    log_jsonl_path = _prepare_output_path(
        None if log_jsonl is None else Path(log_jsonl),
        description="box-picking telemetry",
    )
    if (
        output_mp4_path is not None
        and log_jsonl_path is not None
        and output_mp4_path.resolve() == log_jsonl_path.resolve()
    ):
        raise ValueError("--output-mp4 and --log-jsonl must use different paths")

    needs_overlay = not headless or output_mp4_path is not None
    cv2 = _cv2() if needs_overlay else None
    if not headless:
        assert cv2 is not None
        _require_highgui(cv2)
    processed_frames = 0
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

    return LiveViewPlan(
        cv2=cv2,
        handoff_ready=handoff_ready,
        handoff_started=handoff_started,
        log_jsonl_path=log_jsonl_path,
        log_stream=log_stream,
        needs_overlay=needs_overlay,
        output_mp4_path=output_mp4_path,
        processed_frames=processed_frames,
        stream_config=stream_config,
        user_cancelled=user_cancelled,
        video_writer=video_writer,
        window_created=window_created,
    )


@dataclass(frozen=True)
class LiveViewOutcome:
    """Legacy compatibility result for a complete watch-and-grab run."""

    processed_frames: Any


@dataclass(frozen=True)
class AlignmentWatchOutcome:
    """Result of camera/perception/alignment only; no grasp was executed."""

    processed_frames: int
    handoff_ready: bool
    user_cancelled: bool


@dataclass(frozen=True)
class PickingFrameObservation:
    """One acquired frame after exactly one perception-facade call."""

    frame: Any
    pose_result: PoseResult
    base_pose: BasePoseDiagnostic | None
    estimator_latency_ms: float


@dataclass
class AlignmentSession:
    """Camera/perception resources exposed as loop-free frame primitives."""

    camera: Any
    metadata: Any
    estimator: Any
    calibration: Calibration
    plan: LiveViewPlan
    color_from_depth: FloatArray | None
    log_stream: TextIO | None
    video_writer: Any
    processed_frames: int
    handoff_ready: bool
    user_cancelled: bool
    headless: bool
    max_frames: int | None
    window_name: str

    def has_frame_budget(self) -> bool:
        """Return whether the entrypoint may acquire another frame."""

        return self.max_frames is None or self.processed_frames < self.max_frames

    def acquire_frame(self) -> Any:
        """Capture exactly one frame; camera/SDK errors stay lower-service-owned."""

        try:
            return self.camera.capture()
        except RuntimeError as exc:
            raise LiveViewUnavailableError(
                f"D435 live capture failed: {exc}"
            ) from exc

    def perceive_frame(self, frame: Any) -> PickingFrameObservation:
        """Call the dependency-neutral box facade exactly once for ``frame``."""

        pose_timestamp_s = time.monotonic()
        estimate_start = time.perf_counter()
        pose_result = perceive_box_pose(
            getattr(frame, "raw_color_bgr", None),
            frame.raw_depth_z16,
            self.metadata.depth_profile.intrinsics,
            self.calibration,
            estimator=self.estimator,
            depth_scale=self.metadata.depth_scale_m,
            sensor_timestamp_ms=frame.depth_timestamp_ms,
            frame_id=frame.depth_frame_number,
            timestamp_s=pose_timestamp_s,
        )
        return PickingFrameObservation(
            frame=frame,
            pose_result=pose_result,
            base_pose=_base_diagnostic_from_pose_result(pose_result),
            estimator_latency_ms=1000.0 * (time.perf_counter() - estimate_start),
        )

    def record_frame(
        self,
        observation: PickingFrameObservation,
        *,
        handoff_ready: bool,
    ) -> None:
        """Record/render one decided frame without choosing loop continuation."""

        self.handoff_ready = bool(handoff_ready)
        frame = observation.frame
        overlay: ImageArray | None = None
        if self.plan.needs_overlay:
            assert self.plan.cv2 is not None
            assert self.color_from_depth is not None
            overlay = draw_live_overlay(
                frame.raw_color_bgr,
                observation.base_pose,
                evidence=self.estimator.last_evidence,
                color_from_depth=self.color_from_depth,
                color_intrinsics=self.metadata.color_profile.intrinsics,
                estimator_latency_ms=observation.estimator_latency_ms,
                cv2_module=self.plan.cv2,
            )
        if self.video_writer is not None:
            assert overlay is not None
            self.video_writer.write(overlay)
        _write_live_record(
            self.log_stream,
            frame_index=self.processed_frames,
            depth_frame_number=frame.depth_frame_number,
            depth_timestamp_ms=frame.depth_timestamp_ms,
            estimator_latency_ms=observation.estimator_latency_ms,
            base_pose=observation.base_pose,
            handoff_ready=self.handoff_ready,
        )

        key = -1
        if not self.headless:
            assert self.plan.cv2 is not None and overlay is not None
            try:
                self.plan.cv2.imshow(self.window_name, overlay)
                key = int(self.plan.cv2.waitKey(1)) & 0xFF
            except Exception as exc:
                raise LiveViewUnavailableError(
                    f"OpenCV box-picking display failed: {exc}"
                ) from exc

        self.processed_frames += 1
        if key in {27, ord("q"), ord("Q")}:
            self.cancel()

    def cancel(self) -> None:
        """Mark operator/interrupt cancellation without owning loop control."""

        self.user_cancelled = True
        self.handoff_ready = False

    def outcome(self) -> AlignmentWatchOutcome:
        """Return the current entrypoint-visible alignment outcome."""

        return AlignmentWatchOutcome(
            processed_frames=self.processed_frames,
            handoff_ready=bool(self.handoff_ready),
            user_cancelled=bool(self.user_cancelled),
        )

    def watch(self, automation: LivePoseAutomation | None) -> AlignmentWatchOutcome:
        """Legacy compatibility wrapper; entrypoints must own their frame loop."""

        return watch_for_alignment(session=self, automation=automation)


def watch_for_alignment(
    *,
    session: AlignmentSession,
    automation: LivePoseAutomation | None,
) -> AlignmentWatchOutcome:
    """Compatibility loop assembled only from the public one-frame primitives."""

    try:
        while session.has_frame_budget():
            frame = session.acquire_frame()
            observation = session.perceive_frame(frame)
            handoff_ready = session.handoff_ready
            if automation is not None:
                handoff_ready = automation.update(
                    observation.base_pose,
                    pose_timestamp_s=observation.pose_result.timestamp_s,
                    now_s=time.monotonic(),
                )
            session.record_frame(observation, handoff_ready=handoff_ready)
            if session.user_cancelled or session.handoff_ready:
                break
    except KeyboardInterrupt:
        session.cancel()

    return session.outcome()


@contextmanager
def open_alignment_session(
    *,
    handoff_ready: Any,
    log_stream: Any,
    processed_frames: Any,
    user_cancelled: Any,
    video_writer: Any,
    window_created: Any,
    plan: LiveViewPlan,
    calibration: Calibration,
    estimator_config: Any,
    fullscreen: Any,
    headless: Any,
    max_frames: Any,
    metadata_context: Any,
    window_name: Any,
) -> Iterator[AlignmentSession]:
    """Open and validate perception resources before any robot initialization.

    The yielded session performs no robot lifecycle calls. Its caller visibly
    owns automation start, alignment-stop evidence, grasp, and teardown.
    """

    log_stream = _open_log(plan.log_jsonl_path)
    try:
        with RealSenseAdapter(plan.stream_config) as camera:
            metadata = camera.session_metadata(**dict(metadata_context))
            _validate_camera_profile(calibration, metadata)
            estimator = ParcelPoseEstimator(
                metadata.depth_profile.intrinsics,
                calibration,
                estimator_config,
            )
            color_from_depth: FloatArray | None = None
            if plan.needs_overlay:
                assert plan.cv2 is not None
                color_from_depth = factory_extrinsics_to_transform(
                    metadata.depth_to_color
                )
            if plan.output_mp4_path is not None:
                assert plan.cv2 is not None
                color_intrinsics = metadata.color_profile.intrinsics
                video_writer = _open_video(
                    plan.output_mp4_path,
                    (color_intrinsics.height, color_intrinsics.width),
                    plan.stream_config.fps,
                    cv2_module=plan.cv2,
                )
            if not headless:
                assert plan.cv2 is not None
                try:
                    plan.cv2.namedWindow(window_name, plan.cv2.WINDOW_NORMAL)
                    window_created = True
                    if fullscreen:
                        plan.cv2.setWindowProperty(
                            window_name,
                            plan.cv2.WND_PROP_FULLSCREEN,
                            plan.cv2.WINDOW_FULLSCREEN,
                        )
                except Exception as exc:
                    raise LiveViewUnavailableError(
                        f"OpenCV could not create the box-picking window: {exc}"
                    ) from exc

            yield AlignmentSession(
                camera=camera,
                metadata=metadata,
                estimator=estimator,
                calibration=calibration,
                plan=plan,
                color_from_depth=color_from_depth,
                log_stream=log_stream,
                video_writer=video_writer,
                processed_frames=processed_frames,
                handoff_ready=handoff_ready,
                user_cancelled=user_cancelled,
                headless=headless,
                max_frames=max_frames,
                window_name=window_name,
            )
    finally:
        if video_writer is not None:
            video_writer.release()
        if log_stream is not None:
            log_stream.close()
        if window_created:
            assert plan.cv2 is not None
            try:
                plan.cv2.destroyWindow(window_name)
            except Exception:
                destroy_all = getattr(plan.cv2, "destroyAllWindows", None)
                if callable(destroy_all):
                    destroy_all()


def watch_and_grab(
    *,
    handoff_ready: Any,
    handoff_started: Any,
    log_stream: Any,
    processed_frames: Any,
    user_cancelled: Any,
    video_writer: Any,
    window_created: Any,
    plan: LiveViewPlan,
    automation: Any,
    calibration: Any,
    estimator_config: Any,
    fullscreen: Any,
    headless: Any,
    max_frames: Any,
    metadata_context: Any,
    window_name: Any,
) -> LiveViewOutcome:
    """Compatibility wrapper preserving the existing lifecycle and trace order."""

    outcome = AlignmentWatchOutcome(
        processed_frames=processed_frames,
        handoff_ready=bool(handoff_ready),
        user_cancelled=bool(user_cancelled),
    )
    try:
        with open_alignment_session(
            handoff_ready=handoff_ready,
            log_stream=log_stream,
            processed_frames=processed_frames,
            user_cancelled=user_cancelled,
            video_writer=video_writer,
            window_created=window_created,
            plan=plan,
            calibration=calibration,
            estimator_config=estimator_config,
            fullscreen=fullscreen,
            headless=headless,
            max_frames=max_frames,
            metadata_context=metadata_context,
            window_name=window_name,
        ) as session:
            if automation is not None:
                automation.start()
            outcome = session.watch(automation)

        if (
            outcome.handoff_ready
            and not outcome.user_cancelled
            and automation is not None
        ):
            handoff_started = True
            automation.handoff()
    except KeyboardInterrupt:
        if handoff_started:
            raise
    finally:
        if automation is not None:
            automation.close()

    return LiveViewOutcome(processed_frames=outcome.processed_frames)


__all__ = [
    "AlignmentSession",
    "AlignmentWatchOutcome",
    "DEFAULT_WINDOW_NAME",
    "LivePoseAutomation",
    "LiveViewUnavailableError",
    "PickingFrameObservation",
    "draw_live_overlay",
    "live_overlay_lines",
    "open_alignment_session",
    "resolve_live_view_plan",
    "watch_and_grab",
    "watch_for_alignment",
]
