# ADR 0002: Opt-in mobile alignment and seamless grasp handoff

## Status

Superseded by
[ADR 0003: Default automatic box-picking entrypoint](0003-default-automatic-picking.md),
2026-07-30.

This ADR is preserved as the historical opt-in `live-view --auto-grab` design.

## Context

The operator observed that the box and robot were physically centered when the
previous nominal base output reported `y=-0.050 m`. The operational grasp pose
is a box-volume center at `x=0.740 m`, corrected `y=0.000 m`. After alignment,
the packaged `parcel_pose.grabbing` start/grab/lift sequence must run
without reconnecting or competing with a live mobile command stream.

The camera-to-base registration still lacks independent ground truth. Robot
motion must therefore remain distinct from the default diagnostic viewer and
must preserve the estimator's explicit `nominal_unverified` state.

The [RB-Y1 SDK controller FAQ](https://rainbowrobotics.github.io/rby1-dev/sdk/trobuleshooting/generated/faq.html)
states that navigation and body controllers do not run as independent
simultaneous commands: priority arbitration preempts or refuses one command,
and true arm/base simultaneity requires one combined whole-body stream.

## Decision

1. Store `[0.000, +0.050, 0.000] m` as an empirical base-translation
   correction in calibration, so display, evaluation, and control use one
   coordinate definition.
2. Keep plain `live-view` perception-only. Enable motion only with
   `--auto-grab`; require `--allow-nominal-registration` while the calibration
   remains unvalidated.
3. Restrict execution to a controller-reported RB-Y1 Model M v1.2. After
   power/servo/control-manager preparation, inspect the calibrated torso/head
   posture with a 1-degree tolerance. Skip motion when it already matches. On
   a valid measured mismatch, send exactly one torso/head-only Joint Position
   command with a fixed 5-second minimum time, require `FinishCode.Ok`, and
   re-read the joints to verify the calibrated posture before continuing.
   Invalid, missing, or non-finite joint state fails closed without a command.
4. Move only the two arms to the operator-provided mobile-ready joint targets
   with a completed Joint Position one-shot. Use 0.2 seconds minimum motion
   time, a finite 0.01-second hold, and a 10-second timeout. Treat SDK handler
   completion with `FinishCode.Ok` as the ready contract; do not add a second
   measured arm-joint tolerance gate. Recheck the fixed torso/head pose before
   creating the mobility stream. Any command or feedback failure disconnects
   without base motion.
5. Use bounded simultaneous XY+yaw proportional velocity control through one
   SDK SE(2) command stream. The current packaged grasp posture fixes the parcel
   long-axis target to base `90 deg mod 180`, which appears at the live
   overlay's `+90/-90` signed seam. Compute yaw error as the shortest
   unoriented-line difference, and add `wz * [box_y, -box_x]` orbit
   feed-forward so turning does not unnecessarily displace the relative box
   centre. Scale translation and yaw together to `0.08 m/s` and `0.10 rad/s`
   caps. A sole 20 Hz sender thread refreshes the latest published command
   with a finite 1.0-second hold and substitutes zero when the producer is
   stale. Startup requires validated SE(2) feedback to reach `Running`, and
   terminal/idle/malformed feedback fails closed. Require current and filtered
   yaw to converge with the centre before arrival, then independently verify
   the final long-axis yaw is within 8 degrees of the 90-degree target. A
   vertical 0-degree or ambiguous-family pose is zeroed and rejected before a
   non-zero command because it belongs to the future vertical-grasp branch.
6. Treat pose loss, timeout, camera failure, stream failure, operator exit, and
   posture/model/yaw mismatch as stop-without-grasp conditions. Timestamp each
   pose before estimation so the stale-frame watchdog includes inference time.
7. Require multi-frame arrival hysteresis. Then permanently latch zero, confirm
   at least three pumped zero sends, and require fresh/ready 20 Hz mobility
   state with low wheel velocity continuously for at least 0.35 seconds while
   the stream remains alive. Join the sole sender, cancel the stream, and
   confirm completion before any arm command.
8. Keep `parcel_pose.grabbing` able to accept the already
   connected/prepared robot. The standalone CLI continues to own its own
   connection, while auto-grab reuses one connection through alignment and
   lift.

## Consequences

- The user can measure estimator latency with `python live_view.py` without any
  robot-side effect.
- Automatic execution is deliberately fail-closed and clearly opt-in.
- No mobility stream exists until any required 5-second torso/head correction
  and the arm-only mobile-ready command have completed with OK feedback. The
  measured camera posture is checked after correction and again after the arm
  command.
- Mobile and grasp commands cannot overlap. RB-Y1 priority arbitration does
  not provide independent simultaneous mobility/body controllers; failed
  sender join or stream cancellation blocks the grasp.
- A base that does not measurably settle within 2 seconds blocks the grasp.
- The +50 mm correction is auditable but does not constitute base validation.
- Orientation is controlled together with translation for the current
  horizontal-grasp posture. A later vertical-grasp posture needs an explicit
  0-degree target and separate grasp branch; it must not share this handoff.

## Verification requirements

- Regression tests prove the +50 mm correction is applied once and survives
  calibration serialization.
- Pure closed-loop tests cover speed/slew bounds, line-angle wrapping, orbit
  feed-forward, simultaneous XY/yaw convergence, outliers, dropout, pose-loss
  stop, timeout, arrival hysteresis, and exactly-once handoff.
- Fake-SDK integration tests prove `zero pump -> measured stop -> sender join ->
  cancel -> wait -> grasp` ordering on the same robot object and prove every
  failure path omits the grasp.
- Builder and fake-SDK tests prove the exact camera/arm targets, component-only
  Joint Position configuration, and `prepare -> optional camera correction ->
  mobile ready -> create stream` ordering. A matched camera posture must skip
  its one-shot; timeout, bad feedback, or failed measured recheck must create
  no mobility stream.
- No verification step may connect to or command a physical robot outside an
  explicit on-robot operator run.
