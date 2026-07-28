# ADR 0002: Opt-in mobile alignment and seamless grasp handoff

## Status

Accepted, 2026-07-28.

## Context

The operator observed that the box and robot were physically centered when the
previous nominal base output reported `y=-0.050 m`. The operational grasp pose
is a box-volume center at `x=0.740 m`, corrected `y=0.000 m`. After alignment,
the existing `Palletizing/grabbing_box.py` start/grab/lift sequence must run
without reconnecting or competing with a live mobile command stream.

The camera-to-base registration still lacks independent ground truth. Robot
motion must therefore remain distinct from the default diagnostic viewer and
must preserve the estimator's explicit `nominal_unverified` state.

## Decision

1. Store `[0.000, +0.050, 0.000] m` as an empirical base-translation
   correction in calibration, so display, evaluation, and control use one
   coordinate definition.
2. Keep plain `live-view` perception-only. Enable motion only with
   `--auto-grab`; require `--allow-nominal-registration` while the calibration
   remains unvalidated.
3. Restrict execution to a controller-reported RB-Y1 Model M v1.2 and verify
   that torso/head remain within 1 degree of the fixed calibration posture
   before opening a command stream.
4. Use bounded XY-only proportional velocity control through one SDK command
   stream. Do not command yaw; require the observed box orientation to be
   within 8 degrees of a supported 0- or 90-degree symmetry axis before grasp.
5. Treat pose loss, timeout, camera failure, stream failure, operator exit, and
   posture/model/yaw mismatch as stop-without-grasp conditions. Timestamp each
   pose before estimation so the stale-frame watchdog includes inference time.
6. Require multi-frame arrival hysteresis. Then send repeated zero commands,
   cancel the stream, confirm stream completion, and require fresh/ready 20 Hz
   mobility state with low wheel velocity continuously for at least 0.35
   seconds before any arm command.
7. Refactor `grabbing_box.py` so its motion sequence accepts the already
   connected/prepared robot. The standalone CLI continues to own its own
   connection, while auto-grab reuses one connection through alignment and
   lift.

## Consequences

- The user can measure estimator latency with `python live_view.py` without any
  robot-side effect.
- Automatic execution is deliberately fail-closed and clearly opt-in.
- Mobile and grasp commands cannot overlap; failed stream cancellation blocks
  the grasp.
- A base that does not measurably settle within 2 seconds blocks the grasp.
- The +50 mm correction is auditable but does not constitute base validation.
- Orientation is observed but not controlled in this phase; an incompatible or
  unavailable yaw blocks the existing grasp.

## Verification requirements

- Regression tests prove the +50 mm correction is applied once and survives
  calibration serialization.
- Pure closed-loop tests cover speed/slew bounds, outliers, dropout, pose-loss
  stop, timeout, arrival hysteresis, and exactly-once handoff.
- Fake-SDK integration tests prove `zero -> cancel -> wait -> grasp` ordering on
  the same robot object and prove every failure path omits the grasp.
- No verification step may connect to or command a physical robot outside an
  explicit on-robot operator run.
