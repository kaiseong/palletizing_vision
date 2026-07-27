# RB-Y1 D435 parcel pose

Perception-only Python package for estimating the center and long-axis yaw of one fixed `400 x 250 x 150 mm` closed parcel from an Intel RealSense D435 mounted on RB-Y1.

The estimator uses calibrated table/top-plane geometry and a fixed-size metric rectangle. It does not generate base, arm, grasp, power, contact, or end-effector commands.

## Coordinate and calibration contract

All transforms are active, column-vector, target-from-source matrices:

```text
p_target = T_target_from_source @ p_source
T_base_from_depth = T_base_from_head @ T_head_from_color @ E_color_from_depth
```

`E_color_from_depth` comes from the active RealSense depth profile via `get_extrinsics_to(color_profile)`. The supplied nominal head-to-color pose is an initialization only. D435 `Depth Start Point = -4.2 mm` is a mechanical front-glass datum; it is not added to raw Z16, metric depth, or an SDK-frame transform chain.

Calibration states are deliberately separate:

- `nominal`: only nominal mounting geometry is available.
- `plane_calibrated_partial`: the empty-table plane is calibrated, but absolute base-plane X/Y/yaw is unverified.
- `base_validated`: an independent base-referenced source validates the full transform.

Without a complete base transform, the package returns depth/table-plane coordinates and does not populate `*_base` fields.

## Geometry pipeline

```text
raw Z16 + raw-depth intrinsics
        -> top-plane depth slab
        -> pixel ray / top-plane intersection
        -> continuous metric plane points
        -> fixed 0.400 x 0.250 m rectangle fit
        -> crop/edge/conditioning observability
        -> center + yaw modulo 180 + confidence
```

The table plane uses `n dot p = d`, with unit `n` oriented toward the camera/box side. The top plane is offset along the physical normal:

```text
p_top = p_table + 0.150 * n
d_top = d_table + 0.150
```

Image borders are censored observations, not box edges. A missing box-axis coordinate or a 90-degree long/short ambiguity is returned as invalid/null with a reason, never as a forced high-confidence point.

## Installation

Target runtime is Python 3.12 on JetPack 6.2.2. Connected-component filtering requires OpenCV; if `cv2` is unavailable, the estimator fails closed with `component_filter_unavailable` and cannot mark geometry or a full pose valid. `pyrealsense2` is lazy-loaded only for live capture and recording.

```bash
cd /home/kgs/workspace/Palletizing/Codex
python3.12 -m pip install -e '.[vision]'
```

The `vision` extra installs a headless PyPI OpenCV build for generic Python 3.12 environments. On Jetson, first check whether the Python 3.12 environment can already import the JetPack-matched OpenCV build; if it can, use `pip install -e .` so pip does not replace that build. Install the RealSense Python binding using the Jetson/librealsense method appropriate to the target image. The package intentionally does not download or install an architecture-specific RealSense wheel automatically.

```bash
python3.12 -c 'import cv2; print(cv2.__version__)'
python3.12 -m pip install -e '.[test,vision]'
```

## Configuration

Start from `configs/d435_rby1_nominal.json`.

Before using base-frame output:

1. Resolve the actual robot frame name (`link_head_2` versus any robot-model alias).
2. Supply the fixed `T_base_from_head` from RB-Y1 FK/state.
3. Record an empty table and generate a table-plane calibration.
4. Verify the transformed table normal and mounting convention.

The default nominal transform uses the provided `[x,y,z,roll,pitch,yaw]` value `[0.049,-0.0115,0.057,-90,0,-90]` with provisional `Rz(yaw) @ Ry(pitch) @ Rx(roll)` interpretation.

## Record raw D435 evidence

Record an empty-table session first, then full/cropped box sessions. Raw streams are authoritative; aligned color-on-depth is optional debug evidence.
The complete Python 3.12 command sequence and required capture matrix are in
[`../RECORDING_GUIDE.md`](../RECORDING_GUIDE.md).

```bash
python record.py --session-name empty_table --duration-sec 10
```

The wrapper fixes raw Depth/RGB at `640x480 @ 30 FPS`, uses the nominal config,
and stores sessions below `../recordings/codex_640x480/`. Advanced users can
still call `python3.12 -m parcel_pose.cli record ...` directly.

Recommended capture set:

- empty table;
- full box at varied XY and yaw;
- combined translation and rotation;
- each one-edge crop and multi-edge crop while the center remains visible;
- repeated stationary bursts;
- representative tape, labels, lighting, and cardboard texture.

Each session preserves raw Z16/RGB, depth scale, both intrinsics/distortion, factory stream extrinsics, timestamps, stream/device/settings metadata, robot state supplied in configuration, box model, and annotations.

## Calibrate the table plane

```bash
PYTHONPATH=src python3.12 -m parcel_pose.cli calibrate-plane \
  --session ../recordings/codex_640x480/empty_table \
  --output ../out/calibrations/table_plane.json
```

The calibration uses deterministic RANSAC and stores global/per-frame residuals,
inlier ratios, normal direction, thresholds, and `quality_passed`. A failed
quality gate aborts calibration without writing an artifact. If the table does
not dominate the full image, set `table_calibration.roi_uv` in the config or
pass a half-open raw-depth ROI, for example `--roi 80 80 560 450`.

## Replay

```bash
PYTHONPATH=src python3.12 -m parcel_pose.cli replay \
  --session ../recordings/codex_640x480/box_pose_01 \
  --calibration ../out/calibrations/table_plane.json \
  --config configs/d435_rby1_nominal.json \
  --output-jsonl ../out/results/box_pose_01.jsonl
```

Add `--burst-size 5 --burst-min-valid 3` to emit every single-frame result plus
a `result_kind=stationary_burst` record after each fresh five-frame window.

## Live perception

```bash
PYTHONPATH=src python3.12 -m parcel_pose.cli live \
  --calibration ../out/calibrations/table_plane.json \
  --config configs/d435_rby1_nominal.json
```

Live output remains perception-only JSON. It captures five frames by default,
emits each single-frame result, then emits a stationary burst result. Set
`--frames`/`--burst-size` to 5-10 for the initial acceptance path, or
`--burst-size 0` for single-frame diagnostics only. Moving-base continuous
visual-servo timing is outside this phase.

## Verification

```bash
python3.12 -m compileall -q src tests
python3.12 -m pytest -q
PYTHONPATH=src python3.12 -m parcel_pose.cli --help
```

Local verification covers synthetic tilted planes, fixed-size fitting, holes/outliers, crops, long/short ambiguity, angle boundaries, transform composition, output safety, raw recording round-trip, deterministic replay, and SDK-free imports.

The default fitter caps deterministic plane support at 6,000 points. This keeps
more geometric support than the faster 4,000-point setting, which proved brittle
on a full-chain synthetic regression. Development-host timing is configuration
evidence, not a Jetson latency claim; profile again on the Orin before selecting
a visual-servo update rate.

The real labeled targets are p95 center error `<=20 mm` and yaw error modulo 180 `<=4 degrees`, but these remain unverified until an independent base-referenced calibration/ground-truth source exists. Unlabeled recordings can establish repeatability, residuals, coverage, and correct abstention only.

See [ADR 0001](docs/adr/0001-metric-top-plane-estimator.md) for the design decision and rejected alternatives.
