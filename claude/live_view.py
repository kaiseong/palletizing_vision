#!/usr/bin/env python3
"""Live box top-edge overlay + pose HUD (x, y, z, yaw) + latency for the D435.

On the robot (Jetson, live camera):
    python live_view.py --live --camera d435

Preview from a recording (no camera needed; same HUD/latency):
    python live_view.py --recording recordings/box_complex --loop

Speed: --scale 0.5 downsamples depth for the estimate (points stay metric, so
the overlay is still drawn crisp on the full-res colour frame); latency is the
per-frame estimate time shown on the HUD.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from box_orient import (  # noqa: E402
    OrientConfig,
    RealsenseSource,
    RecordingSource,
    camera_to_base_static,
    camera_to_t5_static,
    estimate_box_orientation,
)
from box_orient.geometry import CameraIntrinsics  # noqa: E402


def parse_box(text: str):
    a, b, c = (float(x) for x in text.lower().replace(" ", "").split("x"))
    return max(a, b) / 1000.0, min(a, b) / 1000.0, c / 1000.0


def scale_inputs(depth_m, intr, scale):
    if scale >= 0.999:
        return depth_m, intr
    h, w = depth_m.shape
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    d2 = cv2.resize(depth_m, (nw, nh), interpolation=cv2.INTER_NEAREST)
    i2 = CameraIntrinsics(intr.fx * scale, intr.fy * scale, intr.cx * scale, intr.cy * scale, nw, nh)
    return d2, i2


def draw(bgr, est, intr, latency_ms, frame_label):
    out = bgr.copy()
    show = est.ok and est.center_camera_m is not None
    if show:
        c = np.asarray(est.center_camera_m, dtype=np.float64)
        long_v = np.asarray(est.long_axis_camera, dtype=np.float64) * (est.long_len_m or 0.0) * 0.5
        short_v = np.asarray(est.short_axis_camera, dtype=np.float64) * (est.short_len_m or 0.0) * 0.5
        corners = np.array([c - long_v - short_v, c + long_v - short_v, c + long_v + short_v, c - long_v + short_v])
        if np.all(corners[:, 2] > 0):
            cv2.polylines(out, [intr.project(corners).astype(np.int32).reshape(-1, 1, 2)], True, (0, 230, 0), 2)
        if c[2] > 0 and (c + long_v)[2] > 0:
            p0 = intr.project(c[None])[0].astype(int)
            p1 = intr.project((c + long_v)[None])[0].astype(int)
            cv2.arrowedLine(out, tuple(p0), tuple(p1), (255, 110, 0), 2, tipLength=0.25)

    if show and frame_label in ("base", "t5") and est.center_ref_m is not None:
        pos = np.asarray(est.center_ref_m, dtype=np.float64)
        fl = "base" if frame_label == "base" else "T5"
    elif show:
        pos, fl = np.asarray(est.center_camera_m, dtype=np.float64), "cam"
    else:
        pos, fl = None, frame_label

    if pos is not None:
        lines = [
            f"x={pos[0] * 1000:+.0f}  y={pos[1] * 1000:+.0f}  z={pos[2] * 1000:+.0f} mm ({fl})",
            f"yaw={est.yaw_deg:+.1f} deg   ref={est.reference}",
            f"latency={latency_ms:.0f} ms",
        ]
    else:
        lines = ["-- no box --", f"latency={latency_ms:.0f} ms"]

    y = 34
    for text in lines:
        cv2.putText(out, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 1, cv2.LINE_AA)
        y += 34
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--live", action="store_true")
    src.add_argument("--recording")
    p.add_argument("--camera", choices=("d405", "d435"), default="d435")
    p.add_argument("--box", type=parse_box, default="400x250x150")
    p.add_argument("--yaw-frame", choices=("base", "t5", "camera"), default="base",
                   help="Frame for the x/y/z/yaw HUD (base=robot base, t5=torso top, camera).")
    p.add_argument("--scale", type=float, default=1.0, help="Depth downsample factor for the estimate (0<scale<=1).")
    p.add_argument("--z-min", type=float, default=0.20)
    p.add_argument("--z-max", type=float, default=1.20)
    p.add_argument("--loop", action="store_true", help="Loop a recording continuously.")
    p.add_argument("--no-window", action="store_true", help="Headless: do not imshow (for latency tests / saving).")
    p.add_argument("--save", help="Save the first rendered frame to this path.")
    p.add_argument("--max-frames", type=int, help="Stop after this many frames (headless tests).")
    return p.parse_args()


def main():
    args = parse_args()
    box_long, box_short, box_height = args.box
    cfg = OrientConfig(z_min=args.z_min, z_max=args.z_max)
    if args.yaw_frame == "base":
        camera_to_ref = camera_to_base_static(args.camera)
    elif args.yaw_frame == "t5":
        camera_to_ref = camera_to_t5_static(args.camera)
    else:
        camera_to_ref = None
    lat_hist = []
    saved = False

    def handle(frame) -> bool:
        nonlocal saved
        depth_s, intr_s = scale_inputs(frame.depth_m, frame.intr, args.scale)
        t0 = time.perf_counter()
        est = estimate_box_orientation(
            depth_s, intr_s,
            box_long_m=box_long, box_short_m=box_short, box_height_m=box_height,
            camera_to_ref=camera_to_ref, config=cfg,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        lat_hist.append(latency_ms)
        overlay = draw(frame.bgr, est, frame.intr, latency_ms, args.yaw_frame)
        if args.save and not saved:
            cv2.imwrite(args.save, overlay)
            saved = True
        if not args.no_window:
            cv2.imshow("box_orient live", overlay)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False
        return not (args.max_frames and len(lat_hist) >= args.max_frames)

    if args.live:
        with RealsenseSource() as cam:
            print(f"# live {args.camera}  fx={cam.intr.fx:.0f}  scale={args.scale}  — q to quit", file=sys.stderr)
            for frame in cam.frames():
                if not handle(frame):
                    break
    else:
        src = RecordingSource(args.recording)
        print(f"# recording {args.recording}  frames={len(src)}  scale={args.scale}", file=sys.stderr)
        running = True
        while running:
            for frame in src.frames():
                if not handle(frame):
                    running = False
                    break
            running = running and args.loop
    if not args.no_window:
        cv2.destroyAllWindows()
    if lat_hist:
        a = np.asarray(lat_hist)
        print(f"# latency ms: mean={a.mean():.0f} median={np.median(a):.0f} p90={np.percentile(a,90):.0f} "
              f"(~{1000/max(a.mean(),1e-6):.1f} fps)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
