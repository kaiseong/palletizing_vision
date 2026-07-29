# ADR 0007: Continuous slot-1 acquisition and vision-gated placement

Status: accepted for supervised slot-1 commissioning

Supersedes ADR 0003 and ADR 0004 where those documents describe hover-only
terminal behavior or stop-step forward acquisition. ADR 0005 and ADR 0006
remain active for standalone loaded-ready startup and demonstrated hole
reference provenance.

## Context

The previous slot-1 runtime could hover at `ARRIVED_HOLD` but did not place the
box. A supervised live retry also showed that repeated `10 mm` stop-step
commands could look choppy and fault on post-brake odometry before the camera
saw the complete centre opening. The current upstream assumption remains the
same: another stage roughly aligns RB-Y1 to the pallet front, and this runtime
starts after the operator has the carton held at the configured slot-1 ready
posture or lets `--ensure-slot1-ready` restore that posture.

The commissioned sensors and calibration are still limited. The D435 and FK
provide metric geometry, but the camera registration remains
`nominal_unverified` with the empirical `+0.050 m` base-y correction. There is
no independent ground truth for absolute pallet placement. F/T feedback is not
part of the current placement decision; every `placement.maximum_force_n`,
`maximum_torque_nm`, force-jump, torque-jump, and load-transfer threshold in
`configs/rby1m_v1_2_pallet_slot1_nominal.json` is `null`.

## Decision

- Keep `pallet.py live` as the operator facade. Full slot-1 execution requires
  the explicit flags `--ensure-slot1-ready`, `--auto-palletize-slot1`,
  `--auto-place-slot1`, `--allow-nominal-registration`,
  `--allow-geometry-only-grip-check`, and
  `--allow-vision-geometry-release`.
- Replace repeated stop-step acquisition with `continuous_forward`: five
  stationary relaxed edge-pair frames authorize one `0.030 m/s` forward cruise,
  with a release-capped `0.150 m` budget and `0.006 m` braking allowance.
- Project D435 depth-optical `+X` onto the fitted stack plane to obtain
  `u_right`, orient the stack-plane normal, and compute
  `v_far = normalize(n_stack x u_right)`. The result must align positively with
  the calibrated approach vector. The active config stores both
  `fixed_approach_v_far_axis_base_xy` and `fixed_approach_axis_source`; a
  missing or ambiguous provenance is invalid.
- Treat a metric outer L-corner plus the measured `660 x 658 mm` outer footprint
  as enough to recover a valid stack SE(2) and centred-opening proxy before the
  centre opening is visible. Complete-hole geometry becomes cross-check and
  refinement evidence, not the sole fine-alignment authority.
- During continuous cruise, a fresh strict L-corner proxy or raw complete-hole
  observation immediately commands zero and waits for wheel stop plus
  stationary dwell before handoff.
- After zero handoff, fine alignment uses coupled x/y/yaw against the
  demonstrated reference: opening/proxy centre `[0.865000, 0.139523] m`, yaw
  `-90 deg`, in `base_at_configured_slot1_ready_pose`.
- The visual dropout bridge may propagate one controller-accepted metric
  observation through fresh odometry for at most `0.30 s`. Odometry-propagated
  samples cannot refresh the bridge; expiry, stale odometry, or conflicting
  geometry selects exact-zero mobility.
- Keep one whole-body stream alive. Each packet contains torso/head Position,
  both arm commands, and mobile SE(2) velocity. The loaded place path uses
  bilateral Cartesian impedance for arm lower/release with nullspace joint
  targets copied from measured arm joints. The mobile command is frozen at
  exact zero once placement begins.
- Placement starts only from `ARRIVED_HOLD` with a current metric alignment
  feature held for `1.0 s`, exact-zero Running feedback, fresh stopped-wheel
  dwell, loaded Cartesian-hold mode, fresh measured FK, and fresh vision
  geometry.
- The previous fixed-distance lowering is rejected. `0.050 m` is now only the
  pre-motion clearance floor. Placement freezes a typed immutable
  `PlacementDescentPlan` from the current FK box-bottom and stack-plane bounds,
  retains that raw gap for clearance validation, then commands `2/3` of the
  gap as the supervised MVP descent. This empirical factor compensates the
  current nominal, externally unverified registration and is not contact proof.
- The lowering command copies the acknowledged loaded-hold target and shifts
  both EEF targets by `2/3` of the frozen raw gap in RB-Y1 base z. It preserves
  orientation, squeeze metadata, and nullspace targets rather than re-basing on
  a compliant measured wrist pose and accidentally ratcheting the squeeze. It
  must reach measured z within `0.008 m`, midpoint XY drift within `0.015 m`,
  and rotation within `3 deg` before any release path is considered.
- Release is vision/FK gated. The sequencer requires at least three fresh gap
  samples, `0.008 m` gap stability, evidence age `<=0.30 s`, plan age
  `<=15.0 s`, FK box-bottom lower bound minus stack-plane upper bound
  `>=0.050 m`, uncertainty `<=0.025 m`, and raw gap `<=0.250 m`. After a
  `0.35 s` geometry-evidence dwell, it cancels the loaded squeeze target and
  spreads an additional `0.120 m` per arm. The release timeout is `12.0 s`.
- Release completion requires Running feedback, each EEF within `0.012 m` and
  `4 deg` of the release target, measured inter-EEF separation increase at
  least `0.216 m`, and a `0.35 s` target dwell.
- F/T values are optional telemetry in this configuration. Missing or invalid
  F/T becomes explicit zero-fallback diagnostics, and no F/T threshold can trip
  because the thresholds are `null`.
- Box-pick lift and pallet Cartesian hold both omit a joint-torque-limit
  override. The RB-Y1 controller therefore applies its per-joint model/runtime
  defaults; the former box-pick-only `[100] * 7 Nm` blanket request is rejected
  because it neither matched the pallet stream nor the controller's per-joint
  default policy.
- `pallet_geometry._dominant_height()` selects the dominant support plane
  inside the configured ROI after held-carton exclusion. This assumes one
  pallet stack in the workspace; a larger staging table or adjacent higher
  stack inside the same ROI can select the wrong plane and must be excluded by
  ROI/site layout.
- Held-carton exclusion requires a held yaw hint. If the runtime cannot provide
  held yaw from FK/EEF geometry, the footprint mask is not constructed and the
  estimator fails closed instead of assuming yaw zero.

## Consequences

The mobile base no longer performs repeated small stop-step moves when the
partial edge evidence is stable. It cruises forward until strict outer-corner
proxy or complete-hole geometry appears, then brakes, verifies zero motion,
collects stationary dwell, and hands off once to the coupled fine x/y/yaw
servo. Once placement starts, later vision drift cannot reopen mobile
authority; the runtime keeps sending exact-zero mobile velocity while the arms
lower and release.

The control stream also avoids a body/mobility ownership gap. Torso/head
Position, arm Cartesian impedance, nullspace hold, and mobile velocity are
sent through one stream at `20 Hz` with a `1.0 s` command hold and `0.05 s`
packet minimum time. The first packet uses Cartesian impedance for both arms
because RB-Y1 binds the stream to that initial controller composition; a later
Joint-to-Cartesian change can report Running without executing arm motion.
Running feedback is therefore only an acknowledgement that the stream accepted
the current component command; target-specific success still comes from
measured FK convergence and dwell gates.

This is not physical validation. The current replay and fake-controller tests
exercise software gates, timing, command construction, and perception
repeatability only. The registration is still `nominal_unverified`, and the
operator must verify physical slot accuracy before treating this as a
commissioned palletizing primitive.

## Verification evidence

The following hardware-free command was run against the current code:

```bash
python pallet.py evaluate \
  --session recordings/codex_640x480/pallet_data \
  --session recordings/codex_640x480/pallet_1_arrived
```

The same recordings were also evaluated directly on the Jetson AGX Orin in
`MAXN` mode after keeping dense ray/point and full-frame mask projections in
`float32` while preserving selected fitting and final metric math in
`float64`. The output reported:

- `pallet_data`: acceptance passed, `302/525` valid frames, latency
  `p50=76.8 ms`, `p95=81.0 ms`.
- `pallet_1_arrived`: acceptance passed, `39/39` valid frames, latency
  `p50=75.8 ms`, `p95=77.0 ms`.
- Against the previous all-`float64` Jetson run, p95 improved by about `10%`
  and `16%`, with unchanged valid counts and acceptance.
- Both sessions: `absolute_placement_accuracy =
  not_measured_no_external_ground_truth`.

## Alternatives rejected

- Continue using repeated `10 mm` stop-step movement: it made live motion
  visibly choppy and exposed post-brake overshoot faults before the hole was
  visible.
- Release from geometry-only lowering: it would spread the hands without a
  fresh seating plan or contact/load evidence.
- Treat F/T as required in this release: the user clarified that F/T is
  effectively unused, and the active config has no finite F/T thresholds.
- Select the highest supported height bin as the stack plane: sparse upper
  clutter in the fixed recordings can be above the true stack top, so the
  current single-stack ROI contract uses the dominant support instead.
