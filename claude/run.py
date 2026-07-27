#!/usr/bin/env python3
"""Estimate single-box yaw (0/90 reference) offline from a recording or live D435.

Examples
--------
Offline replay of a recorded session, save debug overlays, print a summary:
    python run.py --recording recordings/box_sweep --yaw-frame ref \
        --z-min 0.4 --z-max 0.9 --save-overlay out/box_sweep --summary

Live D435 on the robot (defaults: --camera d435, --box 400x250x150):
    python run.py --live --yaw-frame ref --smooth 5
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from box_orient import (  # noqa: E402
    OrientConfig,
    RecordingSource,
    RealsenseSource,
    camera_to_t5_static,
    draw_overlay,
    estimate_box_orientation,
)


def parse_box(text: str) -> tuple[float, float, float]:
    parts = text.lower().replace(" ", "").split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("box must be LxWxH in mm, e.g. 400x250x150")
    long_mm, short_mm, height_mm = (float(p) for p in parts)
    long_mm, short_mm = max(long_mm, short_mm), min(long_mm, short_mm)
    return long_mm / 1000.0, short_mm / 1000.0, height_mm / 1000.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--recording", help="Path to a box-perception-recording-v2 session directory.")
    src.add_argument("--live", action="store_true", help="Use a live RealSense camera.")

    p.add_argument("--camera", choices=("d405", "d435"), default="d435")
    p.add_argument("--box", type=parse_box, default="400x250x150", help="Box LxWxH in mm (default 400x250x150).")
    p.add_argument("--yaw-frame", choices=("camera", "ref"), default="camera",
                   help="'ref' expresses yaw in the robot T5 frame via the static extrinsic.")
    p.add_argument("--head-pitch", type=float, default=None, help="head_1 pitch [rad] for the ref transform.")
    p.add_argument("--z-min", type=float, default=0.20)
    p.add_argument("--z-max", type=float, default=1.50)
    p.add_argument("--smooth", type=int, default=1, help="Temporal window (frames) for yaw smoothing.")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=None)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--save-overlay", help="Directory to write per-frame debug overlays.")
    p.add_argument("--summary", action="store_true", help="Print an aggregate summary at the end (offline).")
    p.add_argument("--quiet", action="store_true", help="Do not print per-frame JSON.")
    return p.parse_args()


def circular_mean_mod180(values_deg: list[float]) -> float:
    if not values_deg:
        return float("nan")
    a = np.deg2rad(np.asarray(values_deg, dtype=np.float64) * 2.0)
    mean = np.arctan2(float(np.mean(np.sin(a))), float(np.mean(np.cos(a)))) / 2.0
    return float(np.degrees(mean) % 180.0)


def circular_spread_mod180(values_deg: list[float]) -> float:
    if len(values_deg) <= 1:
        return 0.0
    center = circular_mean_mod180(values_deg)
    def dist(x: float) -> float:
        d = abs((x - center) % 180.0)
        return min(d, 180.0 - d)
    return float(max(dist(v) for v in values_deg))


def main() -> int:
    args = parse_args()
    box_long, box_short, box_height = args.box
    cfg = OrientConfig(z_min=args.z_min, z_max=args.z_max)
    camera_to_ref = None
    if args.yaw_frame == "ref":
        camera_to_ref = camera_to_t5_static(args.camera, head1_pitch_rad=args.head_pitch)

    overlay_dir = None
    if args.save_overlay:
        overlay_dir = Path(args.save_overlay)
        overlay_dir.mkdir(parents=True, exist_ok=True)

    from box_orient.orientation import classify_0_90

    yaw_history: list[float] = []
    raw_yaws: list[float] = []
    ok_count = 0
    total = 0

    def handle(frame) -> None:
        nonlocal ok_count, total
        est = estimate_box_orientation(
            frame.depth_m, frame.intr,
            box_long_m=box_long, box_short_m=box_short, box_height_m=box_height,
            camera_to_ref=camera_to_ref, config=cfg,
        )
        total += 1
        record = {"frame": frame.index, **est.to_dict()}

        if est.yaw_raw_mod180 is not None and not np.isnan(est.yaw_raw_mod180):
            raw_yaws.append(float(est.yaw_raw_mod180))
            yaw_history.append(float(est.yaw_raw_mod180))
            if args.smooth > 1:
                window = yaw_history[-args.smooth :]
                smoothed = circular_mean_mod180(window)
                ref_s, dev_s, yaw_s = classify_0_90(smoothed, cfg.boundary_deg)
                record["smoothed"] = {
                    "reference": ref_s,
                    "deviation_deg": round(dev_s, 2),
                    "yaw_deg": round(yaw_s, 2),
                    "window": len(window),
                }
        if est.ok:
            ok_count += 1

        if not args.quiet:
            print(json.dumps(record, ensure_ascii=False))
        if overlay_dir is not None:
            import cv2

            overlay = draw_overlay(frame.bgr, est, frame.intr)
            cv2.imwrite(str(overlay_dir / f"frame_{frame.index:06d}.jpg"), overlay)

    if args.live:
        with RealsenseSource(serial_number=None) as cam:
            print(f"# live {args.camera} intr fx={cam.intr.fx:.1f} — Ctrl-C to stop", file=sys.stderr)
            try:
                for frame in cam.frames():
                    handle(frame)
            except KeyboardInterrupt:
                pass
    else:
        source = RecordingSource(args.recording)
        print(f"# recording {args.recording}  camera='{source.camera_name}'  frames={len(source)}", file=sys.stderr)
        for frame in source.frames(start=args.start, stop=args.stop, step=args.step):
            handle(frame)

    if args.summary:
        summary = {
            "frames": total,
            "ok_frames": ok_count,
            "ok_rate": round(ok_count / total, 3) if total else 0.0,
            "yaw_median_mod180": round(circular_mean_mod180(raw_yaws), 2) if raw_yaws else None,
            "yaw_spread_deg": round(circular_spread_mod180(raw_yaws), 2) if raw_yaws else None,
        }
        if raw_yaws:
            ref, dev, yaw = classify_0_90(circular_mean_mod180(raw_yaws), cfg.boundary_deg)
            summary["reference"] = ref
            summary["deviation_deg"] = round(dev, 2)
            summary["yaw_deg"] = round(yaw, 2)
        print("# SUMMARY " + json.dumps(summary, ensure_ascii=False), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
