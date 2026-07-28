# RB-Y1 D435 parcel pose

Python package for estimating the center and long-axis yaw of one folded parcel
family from an Intel RealSense D435 mounted on RB-Y1. Eight
physical samples measured `395-401 x 252-256 x 156-164 mm`; the estimator uses
their component-wise median `400 x 253 x 160 mm` as its fixed metric prior.

The estimator core uses calibrated table/top-plane geometry and a fixed-size
metric rectangle and remains perception-only. The `live-view` command has a
separate, explicit `--auto-grab` mode that can stream the RB-Y1 M mobile base
and hand the same connection to the packaged `parcel_pose.grabbing` sequence.

## Coordinate and calibration contract

All transforms are active, column-vector, target-from-source matrices:

```text
p_target = T_target_from_source @ p_source
T_base_from_depth = T_base_from_head @ T_head_from_color @ E_color_from_depth
T_corrected_base_from_depth.translation += [0.000, +0.050, 0.000] m
```

`E_color_from_depth` comes from the active RealSense depth profile via `get_extrinsics_to(color_profile)`. The supplied nominal head-to-color pose is an initialization only. D435 `Depth Start Point = -4.2 mm` is a mechanical front-glass datum; it is not added to raw Z16, metric depth, or an SDK-frame transform chain.

The tracked fixed-setup calibration applies an empirical `+0.050 m` base-y
output correction because the operator observed physical robot/box centering
when the previous nominal output was `y=-0.050 m`. Thus that same observation
now displays and controls as `y=0.000 m`. This is a single-pose visual
correction, not independent base calibration, so the artifact remains
`plane_calibrated_partial` / `nominal_unverified`.

Calibration states are deliberately separate:

- `nominal`: only nominal mounting geometry is available.
- `plane_calibrated_partial`: the empty-table plane is calibrated, but absolute base-plane X/Y/yaw is unverified.
- `base_validated`: an independent base-referenced source validates the full transform.

Without a complete base transform, the package returns depth/table-plane coordinates and does not populate `*_base` fields.
When base registration is validated, `top_center_base_xyz_m` names the fitted
top-surface center and `box_center_base_xyz_m` names the physical volume center
80 mm below it along the table normal; the two z meanings are never conflated.

## Geometry pipeline

```text
raw Z16 + raw-depth intrinsics
        -> top-plane depth slab
        -> pixel ray / top-plane intersection
        -> continuous metric plane points
        -> fixed measured-median 0.400 x 0.253 m rectangle fit
        -> crop/edge/conditioning observability
        -> center + yaw modulo 180 + confidence
```

The table plane uses `n dot p = d`, with unit `n` oriented toward the camera/box side. The top plane is offset along the physical normal:

```text
p_top = p_table + 0.160 * n
d_top = d_table + 0.160
```

The empty table must be visible when this plane is calibrated, but table pixels
do not have to remain visible in every runtime frame. Runtime estimation uses
the stored plane plus the fixed 160 mm height and the observed box top/rim. The
assumption remains valid while the camera/head/torso mount, table height and
tilt, and planar base attitude remain unchanged; changing any of them requires
recalibration.

Image borders are censored observations, not box edges. A missing box-axis coordinate or a 90-degree long/short ambiguity is returned as invalid/null with a reason, never as a forced high-confidence point.

The eight raw measurements and their derived mean, sample standard deviation,
and observed range are preserved in `configs/d435_rby1_nominal.json`. Long and
short size are not re-estimated per frame: their 6/4 mm population spread is
smaller than the D435 edge uncertainty and freeing size would make crop-induced
scale jitter observable as pose motion. Likewise, per-box height adaptation is
not enabled without an explicit `box on table + arms clear` lifecycle signal,
because RGB-D top height alone cannot distinguish a taller box from a lifted
shorter box.

## Installation

Target runtime is Python 3.12 on JetPack 6.2.2. Connected-component filtering requires OpenCV; if `cv2` is unavailable, the estimator fails closed with `component_filter_unavailable` and cannot mark geometry or a full pose valid. `pyrealsense2` is lazy-loaded only for live capture and recording.

```bash
cd /home/kgs/workspace/Palletizing/Codex
# Generic/headless development host only:
python3.12 -m pip install -e '.[test,vision]'
```

The `vision` extra installs a headless PyPI OpenCV build for generic Python 3.12
development hosts. Do not use that command for the Jetson live viewer because
headless OpenCV cannot open an `imshow` window.

On Jetson, retain or install a GUI-capable OpenCV build and use the Python from
the activated environment. A user-level `python3.12` can appear before conda in
`PATH`, so verify the executable explicitly:

```bash
conda activate lerobot
cd ~/kgs_ws/palletizing_vision/Codex
python -c 'import sys, cv2; print(sys.executable); print(cv2.__version__, cv2.currentUIFramework())'
python -m pip install -e '.[test]'
```

`live-view` needs a non-empty UI framework in the first command. A blank value
means that the active Python 3.12 environment still has a headless OpenCV
build. The package intentionally does not install a replacement GUI OpenCV
wheel automatically because that can replace the JetPack-matched build.

The public ARM64 `pyrealsense2` wheel may require a newer glibc than JetPack
6.2.2 provides. Build the official v2.58.3 Python binding inside this repository
instead of replacing JetPack system libraries or the conda environment:

```bash
bash scripts/build_jetson_pyrealsense2.sh
PYTHONPATH=src python -c 'import pyrealsense2 as rs; print(rs.__file__)'
```

The build is self-contained under `.runtime/`; the resulting extension is
copied to `src/` and both paths are ignored by Git. The script does not use
`sudo` or install into `/usr` or conda.

## Sharing only the Codex implementation

Copying the whole `Codex/` directory is the safest distribution unit. The
minimum source set for live pose tracking and `--auto-grab` is:

```text
Codex/
├── src/parcel_pose/    # estimator, live loop, mobile servo, packaged grasp
├── configs/            # D435 model and fixed RB-Y1 calibration artifacts
├── live_view.py        # short live/auto-grab entrypoint
└── pyproject.toml      # Python 3.12 package/install metadata
```

The root `Palletizing/grabbing_box.py`, `robot.py`, `claude/`, recordings, and
generated `out/` videos are not runtime dependencies. `tests/`, `docs/`, and
most of `scripts/` are optional verification/development material. Keep
`scripts/build_jetson_pyrealsense2.sh` only when the target Jetson still needs
the RealSense Python binding built locally. The comparison-only
`scripts/render_algorithm_comparison.py` is intentionally not standalone
because it also loads the Claude implementation.

Python packages and hardware bindings are installed on the robot PC rather
than copied as project folders: NumPy, GUI-capable OpenCV, `pyrealsense2`, and
for `--auto-grab`, `rby1_sdk`. The bundled nominal calibration is tied to the
recorded D435 serial and fixed torso/head pose; distribute a replacement JSON
inside `configs/` after a different camera or mount is calibrated.

## Configuration

Start from `configs/d435_rby1_nominal.json`.

The default raw-Depth table calibration ROI is `[120, 190, 420, 360]`. It was
validated against the fixed RB-Y1/D435/table view in the supplied recordings
and deliberately excludes the background and right-side robot structure. Refit
the ROI after any camera, head, torso, or table movement.

Before using base-frame output:

1. Resolve the actual robot frame name (`link_head_2` versus any robot-model alias).
2. Supply the fixed `T_base_from_head` from RB-Y1 FK/state.
3. Record an empty table and generate a table-plane calibration.
4. Verify the transformed table normal and mounting convention.

The default nominal transform uses the provided `[x,y,z,roll,pitch,yaw]` value `[0.049,-0.0115,0.057,-90,0,-90]` with provisional `Rz(yaw) @ Ry(pitch) @ Rx(roll)` interpretation.

For the fixed recording pose on the confirmed RB-Y1 **M v1.2**, the supplied
torso/head joints and SDK-FK result are stored in
`configs/rby1m_v1_2_fixed_pose.json`. Pass that file to future recordings with
`--robot-state-json`. Because the camera mount is still a nominal seed and no
independent ground truth exists, post-hoc base coordinates from this transform
remain `nominal_unverified`; they are never relabelled as absolute/validated.

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
  --robot-state-json configs/rby1m_v1_2_fixed_pose.json \
  --output ../out/calibrations/table_plane_with_fk.json
```

The calibration uses deterministic RANSAC and stores global/per-frame residuals,
inlier ratios, normal direction, thresholds, and `quality_passed`. A failed
quality gate aborts calibration without writing an artifact. If the table does
not dominate the full image, set `table_calibration.roi_uv` in the config or
pass a half-open raw-depth ROI, for example `--roi 80 80 560 450`.
`--robot-state-json` is also the explicit post-recording FK override for older
sessions whose manifest contains null robot state; it does not promote a
nominal camera mount to `base_validated`.

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

## Performance video

Render every recorded frame on raw RGB while measuring estimator latency and
pose availability:

```bash
PYTHONPATH=src python3.12 -m parcel_pose.cli evaluate-video \
  --session ../recordings/codex_640x480/box_complex \
  --calibration ../out/calibrations/table_plane_with_fk.json \
  --config configs/d435_rby1_nominal.json \
  --output-mp4 ../out/box_complex_base_pose.mp4 \
  --output-summary ../out/box_complex_summary.json \
  --output-jsonl ../out/box_complex_poses.jsonl
```

The displayed `box center base [m]` is the physical box-volume center. The
fitted rectangle lies on the top plane, so the evaluator moves its center
downward by half the representative height (`80 mm`) along the calibrated table
normal.
The video preserves the recording's total timestamp duration and distinguishes
validated registration from FK-plus-nominal-mount output. Without independent
ground truth, its summary reports availability, continuity, and latency—not
center/yaw accuracy.

## Estimator-only performance gate

Use the benchmark runner when changing hot-path geometry. It preloads all depth
frames, runs one full warmup, measures five full passes of the estimator only,
and refuses comparisons across different fixtures, runtime environments, or
settings. Capture the baseline before editing the estimator:

```bash
env PYTHONPATH=src PYTHONHASHSEED=0 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
taskset -c 2-5 python3.12 -m parcel_pose.benchmark capture \
  --session ../recordings_/codex_640x480/box_complex \
  --calibration ../out/box_complex_eval/calibration_fk_nominal.json \
  --config configs/d435_rby1_nominal.json \
  --warmup-passes 1 --repeats 5 \
  --output ../out/perf/baseline.json
```

Run the candidate under exactly the same power, affinity, dependency, and
thermal conditions:

```bash
env PYTHONPATH=src PYTHONHASHSEED=0 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
taskset -c 2-5 python3.12 -m parcel_pose.benchmark compare \
  --baseline ../out/perf/baseline.json \
  --session ../recordings_/codex_640x480/box_complex \
  --calibration ../out/box_complex_eval/calibration_fk_nominal.json \
  --config configs/d435_rby1_nominal.json \
  --warmup-passes 1 --repeats 5 \
  --output ../out/perf/candidate.json
```

The default gate requires identical availability, abstention, observability,
and canonical decisions; center/yaw/confidence must remain within tight numeric
tolerances. It also requires at least 8% better p50, 10% better p95, and 8%
higher estimator throughput. A timing comparison does not establish physical
accuracy because `box_complex` has no independent ground truth.

On the development host (x86_64, Python 3.13.12), direct scalar sorting in the
fixed-window search changed p50 from `112.29` to `79.83 ms`, p95 from `117.26`
to `82.12 ms`, and throughput from `8.97` to `12.55 FPS`, with zero numeric or
decision differences across all 547 frames. These are not Jetson measurements.
Capture a fresh baseline on the Orin with a fixed `nvpmodel`/clock/thermal state
before using the result to select a servo update rate.

## Live perception

```bash
PYTHONPATH=src python3.12 -m parcel_pose.cli live \
  --calibration ../out/calibrations/table_plane.json \
  --config configs/d435_rby1_nominal.json
```

Live output remains perception-only JSON. It captures five frames by default,
emits each single-frame result, then emits a stationary burst result. Set
`--frames`/`--burst-size` to 5-10 for the initial acceptance path, or
`--burst-size 0` for single-frame diagnostics only. This JSON command never
controls the robot; opt-in motion exists only on `live-view`.

## Real-time visualized perception on Jetson

Run the continuous viewer from the Jetson desktop session with the D435
connected over USB 3:

```bash
conda activate lerobot
cd ~/palletizing_vision/Codex
python live_view.py
```

`python3.12 live_view.py` is also accepted: the wrapper re-executes the active
conda interpreter when a user-level Python shadows it in `PATH`.

The wrapper uses the tracked fixed-setup artifact
`configs/rby1m_v1_2_fixed_table_nominal.json` and the nominal estimator config
by default. It also refuses a D435 serial that does not match that calibration.
Override either path when a newly fitted or independently validated calibration
is available:

```bash
python live_view.py \
  --calibration ../out/calibrations/table_plane_with_fk.json \
  --config configs/d435_rby1_nominal.json
```

The native RGB window shows only the fitted four-edge top rectangle and these
five fields:

```text
x=... m   y=... m   z=... m
yaw=... deg   latency=... ms
```

`x/y/z` are the physical box-volume center in the RB-Y1 base frame; the fitted
top center is shifted down by 80 mm along the calibrated table normal. `yaw` is
the signed long-axis line angle in `[-90, 90)` degrees. `latency` measures only
`ParcelPoseEstimator.estimate()` on the host, matching the saved evaluation
video; camera blocking and GUI rendering are excluded. If a frame is
underconstrained, x/y/z/yaw display `--`, and prior-frame rectangle evidence is
not reused.

Press `q`, `Q`, `Esc`, or `Ctrl-C` to exit. Useful runtime options are
`--fullscreen`, `--max-frames N`, `--warmup-frames N`, and `--window-name NAME`.
The viewer uses raw RGB plus the active D435 Depth-to-RGB factory extrinsic and
therefore disables the unused color-to-depth aligned image copy. The numeric
pose remains depth-derived; during robot motion, the D435 rolling-shutter RGB
image can momentarily misalign with the projected edge even when the numeric
pose is current.

The default `rby1m_v1_2_fixed_table_nominal.json` is explicitly
`nominal_unverified`. The command warns once on stderr and displays its base
coordinates for diagnostics. Plain `python live_view.py` never imports the
RB-Y1 SDK, connects to the robot, or creates a command stream.

## Opt-in mobile alignment and automatic grasp

The following command **moves the real RB-Y1**. It is intentionally blocked
unless both the automatic mode and acceptance of the current nominal
registration are explicit:

```bash
python live_view.py \
  --auto-grab \
  --allow-nominal-registration
```

The default controller address is `192.168.30.1:50051`; override it with
`--robot-address HOST:PORT`. The path is fixed to RB-Y1 Model M v1.2 and fails
closed on a different reported model/version. Before opening the mobility
stream it also verifies the calibrated fixed posture within 1 degree: torso
`[0, 55, -59.988, 6.532, 0, 0]` degrees and head `[0, 49.846]` degrees.
After robot preparation and before the mobility stream exists, both arms move
with a 0.5-second minimum time under an arm-only Joint Position command to the mobile-ready
posture below.
Torso and head are omitted from this command, then rechecked, so the fixed
camera transform is preserved.

```text
right [6.644, -21.489, -17.252, -129.031, -83.302, 53.394, 37.071] deg
left  [6.644,  21.488,  17.245, -129.036,  83.304, 53.392, -37.070] deg
```

The ready command uses a 0.5-second final hold and a 10-second command timeout.
Completion means that the SDK handler finished and returned `FinishCode.Ok`;
there is no second joint-state tolerance check. Failure, timeout, or
interruption closes the connection without creating a mobility stream.

The automatic path uses the corrected box-volume center and performs:

```text
connect and validate RB-Y1 M v1.2 + fixed torso/head
  -> prepare power/servos/control manager
  -> arm-only mobile-ready Joint Position one-shot + OK feedback
  -> create a zeroed mobility stream
  -> acquire 3 valid poses
  -> require horizontal canonical family (reference=90 deg)
  -> publish simultaneous XY+yaw commands toward x=0.740 m, y=0.000 m
     and horizontal long-axis yaw=90 deg modulo 180
  -> sole 20 Hz mobility pump keeps the SDK stream alive
  -> stable arrival hold
  -> latch zero and acknowledge 3 pumped zero commands
  -> measured mobility-joint stop confirmation while pumping zero
  -> stream cancel + completion wait
  -> existing all-body start_pose -> Cartesian impedance grab
     -> Cartesian impedance lift
```

The current automatic mode is intentionally tied to the existing horizontal
grasp posture. Under the calibrated transform, a visually horizontal box long
axis is base yaw `90 deg mod 180`; the live overlay may show either `+90` or
`-90` because those values are the same unoriented line and meet at the signed
display seam. A future vertical grasp posture should use a separate `0 deg`
target/motion branch rather than weakening this handoff condition.

The mobile command uses simultaneous proportional XY and 180-degree-symmetric
yaw control. It adds the orbit feed-forward `wz * [box_y, -box_x]` so base
rotation does not make the stationary box centre drift unnecessarily in base
coordinates. Linear and angular demands are scaled together to caps of
`0.08 m/s` and `0.10 rad/s`; controller slew limits are `0.15 m/s^2` and
`0.20 rad/s^2`. A three-frame median/line-medoid filter, 30 mm centre jump
gate, and 15 degree line-yaw jump gate reject isolated pose jumps. Vision only
publishes the latest command; one dedicated sender thread is the sole SDK
stream writer and refreshes it at 20 Hz with a finite 1.0-second control hold.
If vision stops publishing for 0.30 seconds, the sender substitutes zero
instead of refreshing a stale non-zero
velocity. This prevents estimator/display stalls from expiring the stream
without allowing a frozen vision loop to keep renewing motion indefinitely.
Startup and each zero acknowledgement require valid component/mobility/SE(2)
feedback with the controller in `Running`; terminal, idle, malformed, or
preempted feedback aborts without grasp.
Pose age starts immediately after D435 capture,
before estimator execution; a result older than 0.30 seconds is rejected
instead of receiving a fresh post-inference timestamp. Missing/invalid/stale
pose sends zero velocity; continuous pose loss for 2 seconds or a 30-second
approach timeout aborts without grasping. Arrival requires both current and
filtered positions inside 10 mm and both current and filtered long-axis yaw
inside 3 degrees of the horizontal 90-degree target, then at least five valid
frames and 0.35 seconds inside 15 mm / 5 degree outer bands. The handoff has an
independent final 8-degree check against that same 90-degree target; a vertical
0-degree or ambiguous-family box is zeroed and rejected before this mode can
move or grasp it. At arrival the pump is permanently zero-latched,
and measured wheel speeds must remain at or below 0.05 rad/s throughout at
least 0.35 seconds of continuous 20 Hz state updates while that zero stream
stays alive. All mobility
joints must report ready and the state stream must remain fresh. Failure to
settle within 2 seconds blocks the arms. After settling, the sender thread is
joined and the mobility stream is cancelled and awaited before any body
command. RB-Y1 controller arbitration does not allow the active navigation
stream and the packaged grasp body one-shots to execute in
parallel: keeping both streams alive could cause the body command to be
refused or silently not execute. The state subscription is stopped before the
grasp FT monitor starts, and the same robot connection is reused throughout.
The mobile-ready Joint Position command is therefore a completed one-shot
before the stream is created. After the stream is released at arrival, the
existing `start_pose` command takes ownership of torso, head, and both arms via
Joint Position control; grab and lift then switch the arms to Cartesian-space
impedance control.

Pressing `q`, `Esc`, or `Ctrl-C` before handoff, a camera/stream exception, a
posture/identity mismatch, unavailable yaw, or failure to confirm stream
cancellation stops/disconnects without calling the packaged grasp. If `Ctrl-C`
or an SDK/feedback exception arrives after an arm command has begun, the
one-shot SDK handler is explicitly cancelled and awaited before the error is
propagated; Ctrl-C cleanup exits with status 130. Partial physical arm motion
may already have occurred.

The packaged grasp motion itself has not been contact-validated
by these camera recordings: it uses a 300 mm inward target, a 150 mm continuing
lift squeeze, 100 Nm per-arm-joint torque limits, and a 100 s lift hold. Its FT
monitor reports force but does not autonomously abort on a threshold. Run the
first physical trials supervised with an accessible emergency stop and tune
those values independently before unattended operation.

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

See [ADR 0001](docs/adr/0001-metric-top-plane-estimator.md) for the estimator
decision and [ADR 0002](docs/adr/0002-opt-in-mobile-auto-grab.md) for the
robot-control boundary and handoff contract.
