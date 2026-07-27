#!/usr/bin/env python3
"""Record synchronized Intel RealSense D405 RGB/depth frames for box-pose analysis."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import numpy as np


FORMAT_VERSION = "box-perception-recording-v2"
VIEW_ROTATIONS = ("none", "cw90", "ccw90", "180")


@dataclass(frozen=True)
class RecordingConfig:
    output_root: str
    session_name: str
    fps: int
    width: int
    height: int
    max_frames: int | None
    duration_sec: float | None
    warmup_frames: int
    rgb_format: str
    depth_format: str
    jpeg_quality: int
    preview: bool
    serial_number: str | None
    align_depth_to_color: bool
    enable_emitter: bool
    laser_power: float | None
    view_rotation: str


@dataclass(frozen=True)
class SessionPaths:
    session_dir: Path
    rgb_dir: Path
    depth_dir: Path
    index_path: Path
    manifest_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record Intel RealSense D405 color frames, aligned depth maps, intrinsics, "
            "timestamps, and per-frame metadata for offline box-pose tests."
        )
    )
    parser.add_argument("--output-root", default="recordings", help="Directory where recording sessions are stored.")
    parser.add_argument("--session-name", help="Session directory name. Defaults to d405_YYYYMMDDTHHMMSSZ.")
    parser.add_argument("--width", type=int, default=1280, help="Color/depth stream width.")
    parser.add_argument("--height", type=int, default=720, help="Color/depth stream height.")
    parser.add_argument("--fps", type=int, default=30, help="Requested RealSense stream FPS.")
    parser.add_argument("--max-frames", type=int, help="Stop after this many saved frames.")
    parser.add_argument("--duration-sec", type=float, help="Stop after this many seconds.")
    parser.add_argument("--warmup-frames", type=int, default=30, help="Frames to wait before saving.")
    parser.add_argument(
        "--rgb-format",
        choices=("npy", "jpg", "png"),
        default="npy",
        help="RGB frame storage format. npy avoids OpenCV dependency in recording environments.",
    )
    parser.add_argument("--depth-format", choices=("npz", "npy"), default="npz", help="Depth map storage format.")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality when --rgb-format jpg is used.")
    parser.add_argument("--preview", action="store_true", help="Show a live preview window. Press q to stop.")
    parser.add_argument("--serial-number", help="Optional RealSense serial number when multiple cameras exist.")
    parser.add_argument(
        "--no-align-depth",
        action="store_true",
        help="Do not align depth to the color frame. Leave this off for box-pose datasets.",
    )
    parser.add_argument(
        "--disable-emitter",
        action="store_true",
        help="Disable the RealSense IR emitter if the device exposes that option.",
    )
    parser.add_argument(
        "--laser-power",
        type=float,
        help="Optional RealSense laser power value. Only applied when the sensor supports it.",
    )
    parser.add_argument(
        "--view-rotation",
        choices=VIEW_ROTATIONS,
        default="none",
        help=(
            "Metadata for the analysis/viewing orientation relative to saved raw frames. "
            "Use cw90 when the camera is mounted 90 degrees counter-clockwise and frames "
            "must be viewed clockwise."
        ),
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> RecordingConfig:
    if args.max_frames is None and args.duration_sec is None:
        raise ValueError("Set at least one stop condition: --max-frames or --duration-sec.")
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive.")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive.")
    if args.duration_sec is not None and args.duration_sec <= 0.0:
        raise ValueError("--duration-sec must be positive.")
    if args.warmup_frames < 0:
        raise ValueError("--warmup-frames must be non-negative.")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be in [1, 100].")

    return RecordingConfig(
        output_root=str(args.output_root),
        session_name=args.session_name or default_session_name(),
        fps=int(args.fps),
        width=int(args.width),
        height=int(args.height),
        max_frames=None if args.max_frames is None else int(args.max_frames),
        duration_sec=None if args.duration_sec is None else float(args.duration_sec),
        warmup_frames=int(args.warmup_frames),
        rgb_format=str(args.rgb_format),
        depth_format=str(args.depth_format),
        jpeg_quality=int(args.jpeg_quality),
        preview=bool(args.preview),
        serial_number=None if args.serial_number is None else str(args.serial_number),
        align_depth_to_color=not bool(args.no_align_depth),
        enable_emitter=not bool(args.disable_emitter),
        laser_power=None if args.laser_power is None else float(args.laser_power),
        view_rotation=str(args.view_rotation),
    )


def default_session_name(now: datetime | None = None) -> str:
    stamp = now or datetime.now(timezone.utc)
    return stamp.astimezone(timezone.utc).strftime("rec_%Y%m%dT%H%M%SZ")


def prepare_session(output_root: str | Path, session_name: str) -> SessionPaths:
    root = Path(output_root)
    session_dir = root / session_name
    if session_dir.exists():
        raise FileExistsError(f"Recording session already exists: {session_dir}")

    rgb_dir = session_dir / "rgb"
    depth_dir = session_dir / "depth"
    rgb_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)
    return SessionPaths(
        session_dir=session_dir,
        rgb_dir=rgb_dir,
        depth_dir=depth_dir,
        index_path=session_dir / "index.jsonl",
        manifest_path=session_dir / "manifest.json",
    )


def build_manifest(
    config: RecordingConfig,
    *,
    camera_info: dict[str, Any],
    intrinsics: dict[str, float],
    started_at: str,
    depth_intrinsics: dict[str, Any] | None = None,
    depth_scale_m_per_unit: float | None = None,
    extrinsics: dict[str, Any] | None = None,
    stream_profiles: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "created_at": started_at,
        "recording": {
            "frame_count": 0,
            "finished_at": None,
            "duration_sec_actual": None,
        },
        "config": asdict(config),
        "camera": camera_info,
        "intrinsics": intrinsics,
        "color_intrinsics": intrinsics,
        "depth_intrinsics": depth_intrinsics,
        "depth_scale_m_per_unit": depth_scale_m_per_unit,
        "extrinsics": extrinsics or {},
        "stream_profiles": stream_profiles or {},
        "data_layout": {
            "index": "index.jsonl",
            "rgb_dir": "rgb",
            "depth_dir": "depth",
            "rgb_frame_pattern": "rgb/frame_000000.<npy|jpg|png>",
            "depth_frame_pattern": "depth/frame_000000.depth.<npz|npy>",
            "depth_units": "meter",
            "depth_aligned_to_color": bool(config.align_depth_to_color),
            "rgb_color_order": "BGR",
            "saved_frame_orientation": "raw_camera",
            "view_rotation_from_raw_to_analysis": config.view_rotation,
        },
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_safe(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_frame(
    paths: SessionPaths,
    *,
    frame_id: int,
    bgr: np.ndarray,
    depth_m: np.ndarray,
    rgb_format: str,
    depth_format: str,
    jpeg_quality: int,
    wall_time: str,
    monotonic_time_sec: float,
    camera_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rgb_name = f"frame_{frame_id:06d}.{rgb_format}"
    depth_name = f"frame_{frame_id:06d}.depth.{depth_format}"
    rgb_path = paths.rgb_dir / rgb_name
    depth_path = paths.depth_dir / depth_name

    image = require_bgr(bgr)
    depth = require_depth(depth_m)

    if rgb_format == "npy":
        np.save(rgb_path, image)
    elif rgb_format == "jpg":
        cv2 = import_cv2_for_image_io()
        ok = cv2.imwrite(str(rgb_path), image, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
        if not ok:
            raise OSError(f"Failed to write RGB frame: {rgb_path}")
    elif rgb_format == "png":
        cv2 = import_cv2_for_image_io()
        ok = cv2.imwrite(str(rgb_path), image)
        if not ok:
            raise OSError(f"Failed to write RGB frame: {rgb_path}")
    else:
        raise ValueError(f"Unsupported rgb_format: {rgb_format}")

    if depth_format == "npz":
        np.savez_compressed(depth_path, depth_m=depth.astype(np.float32, copy=False))
    elif depth_format == "npy":
        np.save(depth_path, depth.astype(np.float32, copy=False))
    else:
        raise ValueError(f"Unsupported depth_format: {depth_format}")

    record: dict[str, Any] = {
        "frame_id": int(frame_id),
        "wall_time": wall_time,
        "monotonic_time_sec": float(monotonic_time_sec),
        "rgb_path": str(Path("rgb") / rgb_name),
        "depth_path": str(Path("depth") / depth_name),
        "image_shape": list(image.shape),
        "depth_shape": list(depth.shape),
        "depth_stats": depth_statistics(depth),
    }
    if camera_metadata:
        record["camera_metadata"] = camera_metadata
    append_index(paths.index_path, record)
    return record


def append_index(index_path: Path, record: dict[str, Any]) -> None:
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(record), sort_keys=True) + "\n")


def depth_statistics(depth_m: np.ndarray) -> dict[str, int | float | None]:
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    valid_count = int(np.count_nonzero(valid))
    total_count = int(depth.size)
    if valid_count == 0:
        return {
            "valid_count": 0,
            "total_count": total_count,
            "valid_fraction": 0.0,
            "min_m": None,
            "max_m": None,
            "mean_m": None,
            "median_m": None,
            "p05_m": None,
            "p95_m": None,
        }
    values = depth[valid]
    return {
        "valid_count": valid_count,
        "total_count": total_count,
        "valid_fraction": float(valid_count / total_count) if total_count else 0.0,
        "min_m": float(np.min(values)),
        "max_m": float(np.max(values)),
        "mean_m": float(np.mean(values)),
        "median_m": float(np.median(values)),
        "p05_m": float(np.percentile(values, 5)),
        "p95_m": float(np.percentile(values, 95)),
    }


def require_bgr(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"bgr image must have shape HxWx3, got {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def require_depth(depth_m: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth_m must have shape HxW, got {depth.shape}")
    return depth


def import_cv2_for_image_io() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "OpenCV is required only for --rgb-format jpg/png or --preview. "
            "Use the default --rgb-format npy if the recording environment has no OpenCV."
        ) from exc
    return cv2


def import_realsense_sdk() -> Any:
    try:
        import pyrealsense2 as rs  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "Failed to import pyrealsense2. In the active Python environment, run "
            "`python -m pip install pyrealsense2`. On Jetson aarch64, use a Python 3.12 "
            "conda env or Python 3.10 venv; Python 3.13 may have no matching "
            "pyrealsense2 wheel. Also install the RealSense runtime/udev packages if "
            "needed: `sudo apt-get install "
            "librealsense2 librealsense2-udev-rules librealsense2-utils librealsense2-dev`."
        ) from exc
    return rs


def open_realsense_pipeline(rs: Any, config: RecordingConfig) -> tuple[Any, Any]:
    pipeline = rs.pipeline()
    rs_config = rs.config()
    if config.serial_number:
        rs_config.enable_device(config.serial_number)
    rs_config.enable_stream(rs.stream.color, config.width, config.height, rs.format.bgr8, config.fps)
    rs_config.enable_stream(rs.stream.depth, config.width, config.height, rs.format.z16, config.fps)

    try:
        profile = pipeline.start(rs_config)
    except RuntimeError as exc:
        raise SystemExit(
            f"Failed to start RealSense stream: {exc}. "
            f"Tried color/depth={config.width}x{config.height}@{config.fps}. "
            "Check USB3 connection, camera ownership, and whether this D405 supports the requested profile."
        ) from exc
    return pipeline, profile


def configure_depth_sensor(rs: Any, profile: Any, config: RecordingConfig) -> dict[str, Any]:
    depth_sensor = profile.get_device().first_depth_sensor()
    applied: dict[str, Any] = {"depth_scale_m_per_unit": float(depth_sensor.get_depth_scale())}

    emitter_option = getattr(rs.option, "emitter_enabled", None)
    if emitter_option is not None and depth_sensor.supports(emitter_option):
        depth_sensor.set_option(emitter_option, 1.0 if config.enable_emitter else 0.0)
        applied["emitter_enabled"] = bool(config.enable_emitter)
    else:
        applied["emitter_enabled"] = None

    laser_power_option = getattr(rs.option, "laser_power", None)
    if config.laser_power is not None:
        if laser_power_option is not None and depth_sensor.supports(laser_power_option):
            depth_sensor.set_option(laser_power_option, float(config.laser_power))
            applied["laser_power"] = float(config.laser_power)
        else:
            applied["laser_power"] = None
            applied["laser_power_warning"] = "unsupported"
    return applied


def realsense_manifest_metadata(rs: Any, profile: Any, depth_settings: dict[str, Any]) -> tuple[dict, dict, dict, dict, dict]:
    device = profile.get_device()
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()

    color_intrinsics = intrinsics_to_dict(color_profile.get_intrinsics())
    depth_intrinsics = intrinsics_to_dict(depth_profile.get_intrinsics())
    camera_info = {
        "camera_backend": "realsense",
        "name": device_info(device, rs, "name"),
        "serial_number": device_info(device, rs, "serial_number"),
        "product_line": device_info(device, rs, "product_line"),
        "firmware_version": device_info(device, rs, "firmware_version"),
        "usb_type_descriptor": device_info(device, rs, "usb_type_descriptor"),
        "depth_settings": depth_settings,
    }
    stream_profiles = {
        "color": stream_profile_to_dict(color_profile),
        "depth": stream_profile_to_dict(depth_profile),
    }
    extrinsics = {
        "depth_to_color": extrinsics_to_dict(depth_profile.get_extrinsics_to(color_profile)),
        "color_to_depth": extrinsics_to_dict(color_profile.get_extrinsics_to(depth_profile)),
    }
    return camera_info, color_intrinsics, depth_intrinsics, extrinsics, stream_profiles


def intrinsics_to_dict(intrinsics: Any) -> dict[str, Any]:
    return {
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "cx": float(intrinsics.ppx),
        "cy": float(intrinsics.ppy),
        "ppx": float(intrinsics.ppx),
        "ppy": float(intrinsics.ppy),
        "model": str(intrinsics.model),
        "coeffs": [float(value) for value in intrinsics.coeffs],
    }


def extrinsics_to_dict(extrinsics: Any) -> dict[str, Any]:
    return {
        "rotation_row_major": [float(value) for value in extrinsics.rotation],
        "translation_m": [float(value) for value in extrinsics.translation],
    }


def stream_profile_to_dict(profile: Any) -> dict[str, Any]:
    return {
        "stream_type": str(profile.stream_type()),
        "format": str(profile.format()),
        "fps": int(profile.fps()),
        "width": int(profile.width()),
        "height": int(profile.height()),
    }


def device_info(device: Any, rs: Any, name: str) -> Any:
    info_key = getattr(rs.camera_info, name, None)
    if info_key is None:
        return None
    try:
        if device.supports(info_key):
            return device.get_info(info_key)
    except RuntimeError:
        return None
    return None


def frame_metadata(frame: Any, rs: Any) -> dict[str, Any]:
    metadata = {
        "frame_number": int(frame.get_frame_number()),
        "timestamp_ms": float(frame.get_timestamp()),
        "timestamp_domain": safe_call(frame, "get_frame_timestamp_domain"),
        "metadata": {},
    }
    for name in ("frame_counter", "frame_timestamp", "sensor_timestamp", "actual_fps", "exposure", "gain", "laser_power"):
        value_key = getattr(rs.frame_metadata_value, name, None)
        if value_key is None:
            continue
        try:
            if frame.supports_frame_metadata(value_key):
                metadata["metadata"][name] = frame.get_frame_metadata(value_key)
        except RuntimeError:
            continue
    return metadata


def safe_call(obj: Any, name: str) -> Any:
    method = getattr(obj, name, None)
    if method is None:
        return None
    try:
        return str(method())
    except RuntimeError:
        return None


def record_realsense(config: RecordingConfig) -> Path:
    rs = import_realsense_sdk()
    paths = prepare_session(config.output_root, config.session_name)
    pipeline, profile = open_realsense_pipeline(rs, config)
    depth_settings = configure_depth_sensor(rs, profile, config)
    align = rs.align(rs.stream.color) if config.align_depth_to_color else None

    started_at = utc_now_iso()
    start_monotonic = time.monotonic()
    frame_count = 0

    try:
        camera_info, intrinsics, depth_intrinsics, extrinsics, stream_profiles = realsense_manifest_metadata(
            rs, profile, depth_settings
        )
        manifest = build_manifest(
            config,
            camera_info=camera_info,
            intrinsics=intrinsics,
            depth_intrinsics=depth_intrinsics,
            depth_scale_m_per_unit=depth_settings["depth_scale_m_per_unit"],
            extrinsics=extrinsics,
            stream_profiles=stream_profiles,
            started_at=started_at,
        )
        write_json(paths.manifest_path, manifest)

        for _ in range(config.warmup_frames):
            pipeline.wait_for_frames()

        print(f"Recording session: {paths.session_dir}")
        print("Press Ctrl-C to stop." + (" Press q in preview window to stop." if config.preview else ""))

        while should_continue(frame_count, time.monotonic() - start_monotonic, config):
            frames = pipeline.wait_for_frames()
            if align is not None:
                frames = align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            bgr = require_bgr(np.asanyarray(color_frame.get_data())).copy()
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_m = depth_raw.astype(np.float32) * float(depth_settings["depth_scale_m_per_unit"])
            save_frame(
                paths,
                frame_id=frame_count,
                bgr=bgr,
                depth_m=depth_m,
                rgb_format=config.rgb_format,
                depth_format=config.depth_format,
                jpeg_quality=config.jpeg_quality,
                wall_time=utc_now_iso(),
                monotonic_time_sec=time.monotonic(),
                camera_metadata={
                    "aligned_depth_to_color": bool(config.align_depth_to_color),
                    "color": frame_metadata(color_frame, rs),
                    "depth": frame_metadata(depth_frame, rs),
                },
            )

            frame_count += 1
            if frame_count == 1 or frame_count % 10 == 0:
                print(f"Saved {frame_count} frames -> {paths.session_dir}")

            if config.preview:
                cv2 = import_cv2_for_image_io()
                cv2.imshow("box_orient recording", bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
    finally:
        duration_sec = time.monotonic() - start_monotonic
        try:
            pipeline.stop()
        finally:
            if config.preview:
                cv2 = import_cv2_for_image_io()
                cv2.destroyAllWindows()

        if paths.manifest_path.exists():
            manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
            manifest["recording"]["frame_count"] = int(frame_count)
            manifest["recording"]["finished_at"] = utc_now_iso()
            manifest["recording"]["duration_sec_actual"] = float(duration_sec)
            write_json(paths.manifest_path, manifest)

    print(f"Done. Saved {frame_count} frames in {paths.session_dir}")
    return paths.session_dir


def should_continue(frame_count: int, elapsed_sec: float, config: RecordingConfig) -> bool:
    if config.max_frames is not None and frame_count >= config.max_frames:
        return False
    if config.duration_sec is not None and elapsed_sec >= config.duration_sec:
        return False
    return True


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main() -> int:
    config = config_from_args(parse_args())
    record_realsense(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
