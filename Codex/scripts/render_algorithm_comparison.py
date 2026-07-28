#!/usr/bin/env python3
"""Render a timestamp-correct, same-input Codex-versus-Claude diagnostic video.

The Claude source tree is imported read-only.  Both estimators receive the same
native D435 depth frame and intrinsic.  Their accepted poses are converted to
the same RB-Y1 base-frame box-volume-center convention for display.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Iterable

import cv2
import numpy as np


CODEX_ROOT = Path(__file__).resolve().parents[1]
PALLETIZING_ROOT = CODEX_ROOT.parent
CODEX_SRC = CODEX_ROOT / "src"
CLAUDE_ROOT = PALLETIZING_ROOT / "claude"
sys.path.insert(0, str(CODEX_SRC))
sys.path.insert(0, str(CLAUDE_ROOT))

from box_orient.geometry import CameraIntrinsics as ClaudeIntrinsics  # noqa: E402
from box_orient.orientation import (  # noqa: E402
    BoxOrientation,
    OrientConfig,
    estimate_box_orientation,
)
from box_orient.viz import draw_overlay as draw_claude_overlay  # noqa: E402
from parcel_pose.calibration import load_calibration, load_json  # noqa: E402
from parcel_pose.estimator import ParcelPoseEstimator  # noqa: E402
from parcel_pose.evaluation import base_pose_from_estimate  # noqa: E402
from parcel_pose.models import Calibration, EstimatorConfig, PoseEstimate  # noqa: E402
from parcel_pose.recording import SessionReader  # noqa: E402
from parcel_pose.transforms import transform_points  # noqa: E402
from parcel_pose.visualization import draw_pose_overlay  # noqa: E402


PANEL_WIDTH = 640
IMAGE_HEIGHT = 480
HEADER_HEIGHT = 36
FOOTER_HEIGHT = 132
PANEL_HEIGHT = HEADER_HEIGHT + IMAGE_HEIGHT + FOOTER_HEIGHT
OUTPUT_FPS = 30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def text(
    image: np.ndarray,
    value: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.55,
    color: tuple[int, int, int] = (245, 245, 245),
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        value,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        value,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def fit_panel_image(image: np.ndarray) -> np.ndarray:
    if image.shape[:2] == (IMAGE_HEIGHT, PANEL_WIDTH):
        return image
    return cv2.resize(image, (PANEL_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)


def line_angle_delta_deg(left: float, right: float) -> float:
    delta = abs((float(left) - float(right)) % 180.0)
    return min(delta, 180.0 - delta)


def signed_line_angle_deg(angle_rad: float) -> float:
    return (math.degrees(float(angle_rad)) + 90.0) % 180.0 - 90.0


def table_normal_base(calibration: Calibration) -> np.ndarray:
    if calibration.table_plane is None or calibration.T_base_from_depth is None:
        raise ValueError("comparison requires a table plane and T_base_from_depth")
    normal = np.asarray(calibration.table_plane.normal, dtype=np.float64)
    if calibration.table_plane.frame in {"depth", calibration.depth_frame}:
        normal = calibration.T_base_from_depth[:3, :3] @ normal
    elif calibration.table_plane.frame not in {"base", calibration.base_frame}:
        raise ValueError(f"unsupported table plane frame: {calibration.table_plane.frame}")
    return normal / max(float(np.linalg.norm(normal)), 1e-12)


def claude_base_pose(
    estimate: BoxOrientation,
    calibration: Calibration,
    *,
    box_height_m: float,
) -> tuple[np.ndarray, float] | None:
    if (
        not estimate.ok
        or estimate.center_camera_m is None
        or estimate.long_axis_camera is None
        or calibration.T_base_from_depth is None
    ):
        return None
    top_center = np.asarray(
        transform_points(estimate.center_camera_m, calibration.T_base_from_depth),
        dtype=np.float64,
    )
    center = top_center - 0.5 * float(box_height_m) * table_normal_base(calibration)
    axis_base = (
        calibration.T_base_from_depth[:3, :3]
        @ np.asarray(estimate.long_axis_camera, dtype=np.float64)
    )
    yaw = signed_line_angle_deg(math.atan2(float(axis_base[1]), float(axis_base[0])))
    return center, yaw


def create_panel(
    overlay: np.ndarray,
    *,
    title: str,
    title_color: tuple[int, int, int],
    status: str,
    status_color: tuple[int, int, int],
    pose_line: str,
    detail_line: str,
    timing_line: str,
    frame_line: str,
) -> np.ndarray:
    panel = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), 22, dtype=np.uint8)
    panel[HEADER_HEIGHT : HEADER_HEIGHT + IMAGE_HEIGHT] = fit_panel_image(overlay)
    text(panel, title, (12, 25), scale=0.68, color=title_color, thickness=2)
    footer_y = HEADER_HEIGHT + IMAGE_HEIGHT
    cv2.rectangle(panel, (0, footer_y), (PANEL_WIDTH, PANEL_HEIGHT), (22, 22, 22), -1)
    text(panel, status, (12, footer_y + 24), scale=0.62, color=status_color, thickness=2)
    text(panel, pose_line, (12, footer_y + 49), scale=0.53)
    text(panel, detail_line, (12, footer_y + 72), scale=0.50)
    text(panel, timing_line, (12, footer_y + 95), scale=0.46, color=(210, 210, 210))
    text(panel, frame_line, (12, footer_y + 118), scale=0.46, color=(210, 210, 210))
    return panel


def status_run(values: Iterable[bool], target: bool) -> int:
    longest = 0
    current = 0
    for value in values:
        if bool(value) is target:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def temporary_video(target: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.stem}.", suffix=".mp4", dir=target.parent
    )
    os.close(descriptor)
    return Path(name)


def video_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"cannot open video writer: {path}")
    return writer


def verify_video(path: Path, expected_frames: int, expected_size: tuple[int, int]) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"cannot reopen generated video: {path}")
        frame_count = int(round(float(capture.get(cv2.CAP_PROP_FRAME_COUNT))))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        ok, first = capture.read()
        if not ok or first.shape[1::-1] != expected_size:
            raise RuntimeError(f"generated video decode/size check failed: {path}")
    finally:
        capture.release()
    if abs(frame_count - expected_frames) > 1:
        raise RuntimeError(
            f"generated frame count mismatch for {path}: {frame_count} vs {expected_frames}"
        )
    return {
        "path": str(path),
        "frame_count": frame_count,
        "fps": fps,
        "width": expected_size[0],
        "height": expected_size[1],
        "duration_sec": frame_count / fps,
        "bytes": path.stat().st_size,
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = args.output_dir / "box_complex_codex_vs_claude.mp4"
    claude_path = args.output_dir / "box_complex_claude_same_input.mp4"
    summary_path = args.output_dir / "box_complex_algorithm_comparison.json"
    targets = (comparison_path, claude_path, summary_path)
    existing = [str(path) for path in targets if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("outputs already exist; pass --overwrite: " + ", ".join(existing))

    reader = SessionReader(args.session)
    frame_count = len(reader)
    if args.max_frames is not None:
        frame_count = min(frame_count, max(1, int(args.max_frames)))
    if frame_count == 0:
        raise ValueError("session contains no frames")

    calibration = load_calibration(args.calibration)
    config_mapping = load_json(args.config)
    estimator_config = EstimatorConfig.from_root_config(config_mapping)
    codex = ParcelPoseEstimator(
        reader.metadata.depth_profile.intrinsics,
        calibration,
        estimator_config,
    )
    depth_intr = reader.metadata.depth_profile.intrinsics
    claude_intr = ClaudeIntrinsics(
        depth_intr.fx,
        depth_intr.fy,
        depth_intr.cx,
        depth_intr.cy,
        depth_intr.width,
        depth_intr.height,
    )
    claude_config = OrientConfig(z_min=0.20, z_max=1.20)

    manifest = json.loads((args.session / "manifest.json").read_text(encoding="utf-8"))
    timestamps = [float(item["depth_timestamp_ms"]) for item in manifest["frames"][:frame_count]]
    duration_ms = timestamps[-1] - timestamps[0] if frame_count > 1 else 0.0
    output_frame_count = max(1, int(round(duration_ms * OUTPUT_FPS / 1000.0)) + 1)
    output_ticks_ms = 1000.0 * np.arange(output_frame_count, dtype=np.float64) / OUTPUT_FPS
    source_times_ms = np.asarray(timestamps, dtype=np.float64) - timestamps[0]
    output_source_indices = np.searchsorted(
        source_times_ms, output_ticks_ms, side="right"
    ) - 1
    output_source_indices = np.clip(output_source_indices, 0, frame_count - 1)
    repeat_counts = np.bincount(output_source_indices, minlength=frame_count)

    temp_comparison = temporary_video(comparison_path)
    temp_claude = temporary_video(claude_path)
    comparison_writer = video_writer(
        temp_comparison, OUTPUT_FPS, (PANEL_WIDTH * 2, PANEL_HEIGHT)
    )
    claude_writer = video_writer(temp_claude, OUTPUT_FPS, (PANEL_WIDTH, PANEL_HEIGHT))

    codex_valid: list[bool] = []
    claude_valid: list[bool] = []
    codex_latency: list[float] = []
    claude_latency: list[float] = []
    codex_centers: list[np.ndarray | None] = []
    claude_centers: list[np.ndarray | None] = []
    codex_yaws: list[float | None] = []
    claude_yaws: list[float | None] = []
    codex_reasons: Counter[str] = Counter()
    claude_reasons: Counter[str] = Counter()
    started = time.perf_counter()

    try:
        for index, frame in enumerate(reader):
            if index >= frame_count:
                break
            image = frame.color_on_depth_bgr
            if image is None:
                if frame.raw_color_bgr.shape[:2] != frame.raw_depth_z16.shape:
                    raise ValueError("comparison requires color_on_depth or equal stream sizes")
                image = frame.raw_color_bgr
            depth_m = frame.raw_depth_z16.astype(np.float32) * reader.metadata.depth_scale_m

            start = time.perf_counter()
            codex_estimate: PoseEstimate = codex.estimate(
                frame.raw_depth_z16,
                depth_scale=reader.metadata.depth_scale_m,
                timestamp_ms=frame.depth_timestamp_ms,
                frame_id=frame.depth_frame_number,
            )
            codex_ms = 1000.0 * (time.perf_counter() - start)
            codex_pose = base_pose_from_estimate(codex_estimate, calibration)
            codex_ok = codex_pose is not None

            start = time.perf_counter()
            claude_estimate = estimate_box_orientation(
                depth_m,
                claude_intr,
                box_long_m=estimator_config.box_model.long_m,
                box_short_m=estimator_config.box_model.short_m,
                box_height_m=estimator_config.box_model.height_m,
                config=claude_config,
            )
            claude_ms = 1000.0 * (time.perf_counter() - start)
            claude_pose = claude_base_pose(
                claude_estimate,
                calibration,
                box_height_m=estimator_config.box_model.height_m,
            )
            claude_ok = claude_pose is not None

            codex_overlay = draw_pose_overlay(
                image,
                codex_estimate,
                evidence=codex.last_evidence,
                intrinsics=depth_intr,
            )
            claude_overlay = draw_claude_overlay(image, claude_estimate, claude_intr)

            codex_confidence = float(codex_estimate.per_field_confidence.get("yaw", 0.0))
            if codex_pose is None:
                codex_status = "ABSTAIN"
                codex_status_color = (0, 160, 255)
                codex_pose_line = "base center: --    yaw: --"
                codex_detail = "reason: " + ", ".join(codex_estimate.reasons[:2])
                codex_center = None
                codex_yaw = None
                codex_reasons.update(codex_estimate.reasons or ("unspecified",))
            else:
                codex_status = "ACCEPTED"
                codex_status_color = (70, 230, 120)
                codex_center = np.asarray(codex_pose.box_center_xyz_m, dtype=np.float64)
                codex_yaw = float(codex_pose.yaw_signed_deg)
                codex_pose_line = (
                    f"base center [m] x={codex_center[0]:+.3f}  "
                    f"y={codex_center[1]:+.3f}  z={codex_center[2]:+.3f}"
                )
                codex_detail = (
                    f"base yaw={codex_yaw:+.1f} deg   yaw confidence={codex_confidence:.2f}"
                )

            if claude_pose is None:
                claude_status = "REJECTED / NO POSE"
                claude_status_color = (0, 160, 255)
                claude_pose_line = "base center: --    yaw: --"
                claude_detail = "reason: " + ", ".join(claude_estimate.reasons[:2])
                claude_center = None
                claude_yaw = None
                claude_reasons.update(claude_estimate.reasons or ("unspecified",))
            else:
                claude_status = "ACCEPTED"
                claude_status_color = (70, 230, 120)
                claude_center, claude_yaw = claude_pose
                claude_pose_line = (
                    f"base center [m] x={claude_center[0]:+.3f}  "
                    f"y={claude_center[1]:+.3f}  z={claude_center[2]:+.3f}"
                )
                claude_detail = (
                    f"base yaw={claude_yaw:+.1f} deg   confidence={claude_estimate.confidence:.2f}  "
                    f"size={1000.0 * float(claude_estimate.long_len_m or 0.0):.0f}x"
                    f"{1000.0 * float(claude_estimate.short_len_m or 0.0):.0f} mm"
                )

            elapsed = max(0.0, (timestamps[index] - timestamps[0]) / 1000.0)
            frame_line = f"frame {index + 1}/{frame_count}  t={elapsed:05.1f}s"
            codex_panel = create_panel(
                codex_overlay,
                title="CODEX: fixed table plane + measured 400x253 fit",
                title_color=(40, 220, 255),
                status=codex_status,
                status_color=codex_status_color,
                pose_line=codex_pose_line,
                detail_line=codex_detail,
                timing_line=f"offline estimator {codex_ms:.1f} ms | no temporal filter | base nominal",
                frame_line=frame_line,
            )
            claude_panel = create_panel(
                claude_overlay,
                title="CLAUDE: per-frame RANSAC + observed minAreaRect",
                title_color=(255, 180, 90),
                status=claude_status,
                status_color=claude_status_color,
                pose_line=claude_pose_line,
                detail_line=claude_detail,
                timing_line=f"offline estimator {claude_ms:.1f} ms | no temporal filter | same base chain",
                frame_line=frame_line,
            )
            comparison_frame = np.hstack((codex_panel, claude_panel))
            for _ in range(int(repeat_counts[index])):
                comparison_writer.write(comparison_frame)
                claude_writer.write(claude_panel)

            codex_valid.append(codex_ok)
            claude_valid.append(claude_ok)
            codex_latency.append(codex_ms)
            claude_latency.append(claude_ms)
            codex_centers.append(codex_center)
            claude_centers.append(claude_center)
            codex_yaws.append(codex_yaw)
            claude_yaws.append(claude_yaw)
            if (index + 1) % 50 == 0 or index + 1 == frame_count:
                print(
                    f"rendered {index + 1}/{frame_count} | "
                    f"Codex {sum(codex_valid)} valid | Claude {sum(claude_valid)} valid",
                    flush=True,
                )
    except BaseException:
        comparison_writer.release()
        claude_writer.release()
        temp_comparison.unlink(missing_ok=True)
        temp_claude.unlink(missing_ok=True)
        raise

    comparison_writer.release()
    claude_writer.release()
    temp_comparison.replace(comparison_path)
    temp_claude.replace(claude_path)

    codex_center_steps: list[float] = []
    claude_center_steps: list[float] = []
    codex_yaw_steps: list[float] = []
    claude_yaw_steps: list[float] = []
    common_center_differences: list[float] = []
    common_yaw_differences: list[float] = []
    for index in range(1, frame_count):
        if codex_centers[index - 1] is not None and codex_centers[index] is not None:
            codex_center_steps.append(
                1000.0 * float(np.linalg.norm(codex_centers[index] - codex_centers[index - 1]))
            )
            codex_yaw_steps.append(
                line_angle_delta_deg(float(codex_yaws[index]), float(codex_yaws[index - 1]))
            )
        if claude_centers[index - 1] is not None and claude_centers[index] is not None:
            claude_center_steps.append(
                1000.0 * float(np.linalg.norm(claude_centers[index] - claude_centers[index - 1]))
            )
            claude_yaw_steps.append(
                line_angle_delta_deg(float(claude_yaws[index]), float(claude_yaws[index - 1]))
            )
    for codex_center, claude_center, codex_yaw, claude_yaw in zip(
        codex_centers, claude_centers, codex_yaws, claude_yaws, strict=True
    ):
        if codex_center is not None and claude_center is not None:
            common_center_differences.append(
                1000.0 * float(np.linalg.norm(codex_center - claude_center))
            )
            common_yaw_differences.append(
                line_angle_delta_deg(float(codex_yaw), float(claude_yaw))
            )

    videos = {
        "comparison": verify_video(
            comparison_path, output_frame_count, (PANEL_WIDTH * 2, PANEL_HEIGHT)
        ),
        "claude": verify_video(
            claude_path, output_frame_count, (PANEL_WIDTH, PANEL_HEIGHT)
        ),
    }
    summary = {
        "schema_version": 1,
        "input": {
            "session": str(args.session),
            "frames": frame_count,
            "recorded_duration_sec": duration_ms / 1000.0,
            "output_fps": OUTPUT_FPS,
            "output_frames": output_frame_count,
            "timeline": "30 FPS zero-order hold from recorded depth timestamps",
            "same_native_depth_frames": True,
            "ground_truth_available": False,
            "box_model_m": estimator_config.box_model.to_dict(),
            "dimension_prior": (
                None
                if estimator_config.box_dimension_prior is None
                else {
                    key: value
                    for key, value in estimator_config.box_dimension_prior.to_dict().items()
                    if key != "samples"
                }
            ),
        },
        "coordinate_convention": {
            "display_frame": calibration.base_frame,
            "center": (
                "box volume center, half the configured measured-prior height "
                "below fitted top center along table normal"
            ),
            "yaw": "long-axis signed line yaw in [-90, 90) degrees",
            "base_registration": "nominal_unverified",
        },
        "codex": {
            "valid_frames": int(sum(codex_valid)),
            "valid_ratio": float(np.mean(codex_valid)),
            "longest_valid_run": status_run(codex_valid, True),
            "longest_invalid_run": status_run(codex_valid, False),
            "failure_reasons": dict(codex_reasons),
            "latency_ms": percentiles(codex_latency),
            "adjacent_center_step_mm": percentiles(codex_center_steps),
            "adjacent_yaw_step_deg": percentiles(codex_yaw_steps),
        },
        "claude": {
            "valid_frames": int(sum(claude_valid)),
            "valid_ratio": float(np.mean(claude_valid)),
            "longest_valid_run": status_run(claude_valid, True),
            "longest_invalid_run": status_run(claude_valid, False),
            "failure_reasons": dict(claude_reasons),
            "latency_ms": percentiles(claude_latency),
            "adjacent_center_step_mm": percentiles(claude_center_steps),
            "adjacent_yaw_step_deg": percentiles(claude_yaw_steps),
        },
        "common_valid": {
            "frames": len(common_center_differences),
            "center_distance_mm": percentiles(common_center_differences),
            "yaw_line_difference_deg": percentiles(common_yaw_differences),
        },
        "videos": videos,
        "processing_wall_sec": time.perf_counter() - started,
        "accuracy_warning": "No independent ground truth; continuity and availability only.",
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
