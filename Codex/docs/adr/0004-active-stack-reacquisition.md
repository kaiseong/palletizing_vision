# ADR 0004: Forward-only stack reacquisition before slot-1 fine servo

Status: accepted for software verification; physical commissioning blocked

Supersedes the ready-transition ownership portion of ADR 0003. The metric
opening model, single combined owner, and hover-only terminal scope remain in
force.

## Context

The upstream orchestrator already performs marker docking and completes the
fixed pallet-ready posture before this runtime starts. At that endpoint the
D435 may not yet see the complete `148 x 149 mm` central opening, so the
opening-only fine servo cannot produce a fully observed slot target. Recorded
frames do contain a metric connected near/front plus image-right-side L-corner
before the opening becomes visible.

Available constraints are D435 RGB-D and intrinsics, factory RGB/depth
registration, fixed ready-pose FK/camera mount, measured `660 x 658 mm` stack
extent, measured carton family, bilateral EEF FK, RB-Y1 odometry, and a roughly
front-aligned upstream endpoint. The upstream marker pose, a full outer
rectangle, external ground truth, contact calibration, and recording odometry
are intentionally not used as final slot-pose sources.

## Decision

- Keep `pallet.py` as the operator facade and the existing complete-opening
  `PalletSlot1Servo` as the sole x/y/yaw fine authority.
- Add a separate `LCornerObservation`. It retains metric plane and two-line
  evidence plus explicit constrained/unconstrained DOFs, but has no stack
  center, hole center, or slot target field.
- Add a pure stop-and-observe acquisition controller. One fresh five-frame
  stationary L gate may authorize one forward step of at most 10 mm and
  0.03 m/s. It can never emit `vy`, `wz`, or reverse motion.
- Measure each authorized step with fresh `T_odom_base`. Lateral or yaw drift,
  stale/nonfinite odometry, visual loss, timeout, no progress, overshoot, or a
  parent interlock failure selects exact zero. Odometry cannot authorize the
  next step.
- Conserve a session-wide forward budget without reset or refund. The shipped
  default is `0.0 m`; this release rejects values above `0.15 m`, while the
  documented absolute design ceiling remains `0.20 m`. Every target reserves
  the configured stopping allowance, and odometry remains monitored through
  zero-command braking and verified wheel stop so coasting is not hidden.
- Require five complete-hole frames spanning at least 0.35 s, then transfer
  authority once at exact zero and measured wheel stop. Coarse authority is
  permanently revoked for that session.
- Reject both the active `GripHandoff` scaffold and an already-released
  `ReadyHoldHandoff` at the runtime facade and controller before any robot
  command. The current box-pick endpoint does not provide a single-stream epoch
  transfer with exact torso/head/control-mode and stream-identity provenance;
  release-first adoption additionally has an unbounded body-hold gap.
- Require a future reviewed integration to use one persistent combined stream
  with an internal owner epoch transfer, or an equivalent atomic/two-phase
  protocol whose successor packet is acknowledged before source release. It
  must also carry calibrated bilateral EEF-to-box transforms.
- Keep one combined body+mobility stream and preserve the existing grip,
  clearance, F/T, freshness, calibration, and shutdown gates.
- Stop at persistent slot-1 hover. Descent, contact, release, retreat, marker
  integration, and slot 2 remain outside this change.

## Alternatives rejected

- Infer a full slot pose from the partial L: physically underconstrained and
  likely to turn crop into false x/y corrections.
- Fit the full `660 x 658 mm` outer rectangle: the carried carton and close
  view crop the necessary boundaries.
- Continue from odometry or a last-good image after visual loss: removes the
  current-geometry motion predicate.
- Put fallback logic inside the fine servo: weakens the authority boundary and
  permits two incompatible observability models to share one controller.
- Reuse the upstream marker pose inside Codex: that camera/algorithm is owned
  by another stage and is not observable at the close placement viewpoint.
- Release the upstream stream before destination adoption: leaves the carried
  box unsupported by a proven controller during an unbounded software gap.

## Consequences

Replay can demonstrate observability, abstention, deterministic decisions, and
the one-way controller handoff; it cannot demonstrate traveled distance because
the supplied recordings contain no odometry. Generated replay rows therefore
distinguish `would_request_step` from `motion_authorized` and never claim
executed travel.

The current recordings estimate only about 29.5–32.0 mm conservative vertical
clearance, below the retained 50 mm gate. Combined with the shipped zero
acquisition budget and nominal/unverified base registration, physical nonzero
motion remains fail-closed. A supervised future commissioning change must
provide fresh clearance, F/T, odometry, and external accuracy evidence; passing
these software checks alone does not authorize placement or descent.

The original post-release ready-adoption acceptance item is explicitly not met
and neither ownership model is exposed as an actuator capability. Replay and
perception remain deliverable; physical commissioning requires the reviewed
atomic bridge and calibration evidence described above.
