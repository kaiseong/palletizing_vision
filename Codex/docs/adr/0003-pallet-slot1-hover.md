# ADR 0003: Metric slot-1 hover with one persistent control owner

Status: partially superseded by ADR 0004 and ADR 0005

ADR 0004 replaces this document's ready-transition ownership path and adds the
partial-L forward-acquisition phase. The metric opening/slot formulation,
single combined owner, and hover-only scope below remain active. ADR 0005 adds
the standalone loaded-slot1-ready commissioning path; the integrated
box-pick-to-pallet handoff remains unavailable.

## Context

RB-Y1 must carry one fixed carton over the first slot of the third pallet layer. The stack is a four-carton pinwheel with a measured `660 x 658 mm` outer boundary and `148 x 149 mm` center opening. D435 RGB-D, intrinsics, factory RGB/depth extrinsics, fixed camera mount, and fixed pallet-ready joints are available; independent base-frame ground truth and calibrated EEF-to-carton contact transforms are not.

The box-pick implementation currently hands control between a mobility-only stream and separate body commands. That ownership pattern cannot keep a loaded body controller and mobile-base controller alive concurrently.

## Decision

- Expose the feature as `pallet.py` / `pallet`, while retaining implementation modules inside `parcel_pose`.
- Recover the stack frame from a metric top plane and fixed-size center-opening rims. Image-space boxes/Hough lines are not the primary estimator. The outer boundary is optional consistency evidence.
- Compute slot 1 from `p_hole + 0.128*u_right + 0.20175*v_far`; branch-lock `u_right` to its image-right projection.
- Use one phase-aware RB-Y1 component command owner. A single command may contain torso/head Position, both-arm Joint Impedance, and SE(2) mobility.
- Send the five-second ready trajectory exactly once with zero mobility. After fresh per-component feedback and one-degree joint tracking, switch to a fixed-rate steady body-hold plus mobility stream.
- Treat grip continuity and vertical clearance as measured gates, not operator booleans. If held-top/EEF evidence cannot establish the lower bound, nonzero mobility remains disabled.
- Use `pallet_1_arrived` only as an operator-eye nominal ready-pose offset seed: mean slot target minus ready EEF midpoint is `[+0.085463, +0.008236] m` in base XY. Label it `nominal_unverified`; it is not external ground truth or evidence of absolute placement accuracy.
- End the MVP at persistent `ARRIVED_HOLD`: body support continues and mobility remains zero. There is no descent, contact, release, or slide.
- Make live motion opt-in twice and label the current `+50 mm` base-y correction `nominal_unverified`.
- Accept live RGB-D only when both stream timestamps share a RealSense
  `GLOBAL_TIME` or `SYSTEM_TIME` domain and both sensor-to-host age and
  post-receipt processing age remain within 200 ms.

## Alternatives rejected

- Pure RGB/image-space fitting: discards available metric constraints and cannot safely distinguish crop from object motion.
- A new independent `pallet_pose` package: duplicates D435/session/transform code and risks convention drift.
- Separate body and mobility streams: violates the controller ownership and expiry requirements.
- Standalone actuation without a verifiable grip handoff: preempts an unknown squeeze controller and cannot certify load continuity. Standalone `pallet.py` therefore defaults to perception-only; actuator integration must preserve a single owner or pass the measured gate.
- Including placement/release in this MVP: would combine alignment uncertainty with contact-manipulation failures.

## Consequences

Recorded sessions can prove camera-frame metric consistency and temporal stability, not absolute placement accuracy. A nominal EEF midpoint is a proxy rather than a calibrated carton center. The first robot run is a supervised zero-mobility grip-transition commissioning step. `ARRIVED_HOLD` intentionally does not return success and close its controller; a successor must acknowledge ownership before normal shutdown.

The present box-pick implementation cannot yet be that predecessor: its final
lift is a finite one-shot command with a different torque policy, after which
the caller closes the robot lifecycle. Consequently integrated takeover from
box-pick remains blocked, not an operator prompt to bypass. ADR 0005 documents
the separate standalone path where the user starts already holding the box at
the configured ready posture and the pallet process becomes the only live
combined owner. Physical commissioning still requires fixed-ready clearance
audit review against the 50 mm conservative vertical-clearance floor plus
finite site-specific F/T plausibility limits.
The injected destination controller must already be connected and identity/FK
validated before the source creates its immutable handoff snapshot; connection
latency is never hidden by weakening the 150 ms handoff-freshness limit. Until
independent ground truth exists, post-hoc base coordinates from this transform
remain `nominal_unverified`; they are never relabelled as absolute/validated.
