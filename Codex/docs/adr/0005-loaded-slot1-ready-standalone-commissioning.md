# ADR 0005: Loaded slot-1-ready standalone commissioning

Status: accepted for supervised commissioning documentation

Supersedes the "automatic palletizing always stops before robot connection"
claim in ADR 0004 and the README. ADR 0007 later supersedes this document's
Joint Impedance stream composition and hover-only terminal scope; its metric
slot model and one combined-owner requirement remain active. ADR 0004's
integrated box-pick-to-pallet handoff rejection remains active.

## Context

The user can start the pallet runtime after the carton is already gripped and
RB-Y1 is at the configured slot-1 ready posture. This is a standalone
commissioning boundary: the previous owner must be stopped and disconnected
before `pallet.py live` starts. The software can verify or restore the fixed
ready posture, but the user remains responsible for physical commissioning and
for confirming that the box is actually held.

The current calibration remains `nominal_unverified`. F/T plausibility limits
are not configured. Direct close-range depth of the carried box top is not
trusted as the grip/clearance source because the held box is near the camera and
can be cropped, occluded by the robot, or depth-noisy.

## Decision

- Expose standalone loaded alignment only through the complete explicit flag
  set: `--ensure-slot1-ready`, `--auto-palletize-slot1`,
  `--allow-nominal-registration`, and `--allow-geometry-only-grip-check`.
- Require `--headless` for SSH/no-display operation when overlays and telemetry
  are written instead of an OpenCV window.
- Let `--ensure-slot1-ready` verify the configured torso, both arms, and head
  within `1 deg`; if needed, it sends one `5 s` all-JointPosition restoration
  before live perception starts. When already ready outside execute mode, it
  preserves the old skip behavior and does not prepare power, energize servos,
  reset the control manager, or send a motion command.
- For `--auto-palletize-slot1`, connect as the sole process, bootstrap the
  already-held configured ready posture, prepare power/servos/control manager
  even when the posture already matches, and start one RB-Y1 component command
  owner.
- Put torso/head Position, both arms Joint Impedance, and SE(2) mobility in the
  same command stream. The first packet uses exact zero mobility, and stale,
  fault, shutdown, or interlock-failed decisions select zero mobility.
- Allow base alignment only. Descent, contact, release, gripper commands,
  sliding, retreat, and power-off remain outside this MVP.
- Treat `--allow-geometry-only-grip-check` as a commissioning-only substitute
  for unconfigured F/T thresholds only when the reviewed config also sets
  `grip_interlock.fixed_ready_geometry_only_commissioning_enabled=true`. It
  keeps fresh joint tracking, dual-EEF FK, EEF separation stability,
  stack-plane, clearance, wheel-stop, odometry, and command-ownership gates
  active. Configs without that policy fail closed even when the CLI flag is
  present.
- Compute the fixed-ready box-bottom clearance proxy from fresh dual-EEF FK,
  the configured nominal EEF-to-box offset, half the configured maximum carton
  height, configured uncertainty, and the observed stack plane. The stack plane
  source must be a complete stack plane or explicit `metric_coarse_l_corner_plane`.
- Keep the actual forward step behind the independent stationary five-frame
  L-corner gate; accepting the coarse plane as a clearance source does not by
  itself authorize a step.
- Reject `--max-frames` in execute mode because the loaded non-daemon owner
  cannot end normally without successor handoff or explicit forced cancellation.
- Keep integrated `GripHandoff` and released `ReadyHoldHandoff` actuator paths
  rejected until a reviewed atomic stream/epoch transfer exists.

## Alternatives rejected

- Treat the old "actuation always blocked before connection" statement as still
  true: rejected because the standalone loaded-ready path now connects and can
  stream when all explicit commissioning flags are present.
- Use direct close-range RGB-D held-top evidence as the default clearance
  source: rejected because near-field crop, robot occlusion, and depth noise can
  make the carried top unreliable.
- Replace F/T plausibility with a blanket operator boolean: rejected because it
  would erase useful measured gates. The geometry-only override is explicit and
  still requires fresh FK, joint, stability, stack-plane, and wheel evidence.
- Reuse the integrated box-pick handoff scaffolds: rejected because they still
  do not provide atomic owner transfer, stream identity, exact body targets,
  control mode, stiffness, damping, torque policy, and acknowledged successor
  packet provenance.

## Consequences

Documentation must distinguish three surfaces:

- perception-only live/replay, which does not command RB-Y1;
- `--ensure-slot1-ready`, which can verify or restore the fixed ready posture
  but does not create a pallet control stream by itself;
- the full standalone commissioning flag set, which can open the combined
  owner and run the base-alignment controller while the user-supervised box
  remains held. Nonzero base packets still require all motion gates and a
  reviewed positive acquisition budget.

The geometry-only clearance path does not prove grip force, contact loss,
carton deformation, calibrated EEF-to-carton contact transforms, or absolute
placement accuracy. Passing these software gates does not authorize descent or
release.

The fixed-ready audit reports about `0.179 m` conservative box-bottom clearance
against the configured `0.050 m` floor. Direct held-top depth remains
discrepant and untrusted for this close-range carried-box case; physical
commissioning remains pending without implying that the software clearance gate
must necessarily fail.
