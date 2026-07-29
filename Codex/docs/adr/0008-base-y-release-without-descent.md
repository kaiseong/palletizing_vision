# ADR 0008: Release slot-1 by opening along base Y without a vertical descent

- Status: accepted
- Date: 2026-07-29
- Supersedes the descent portion of [ADR 0007](0007-continuous-slot1-place.md)

## Context

Physical commissioning run 06 aligned correctly and reached
`placement=LOWERING:waiting_for_measured_planned_descent`, then the RB-Y1
controller rejected the streamed Cartesian target and the combined stream
reported `This command stream is expired`.  The torso and head are frozen at
the slot-1 ready pose for the whole session, so lowering both wrists in base Z
forces the arms to reach down and outward toward their reach singularity.  The
planned descent was `gap * 2/3`, which for a nominal 90 mm gap commands 60 mm
of wrist travel on both arms simultaneously.

After the stream expired, `ensure_persistent_zero_body_hold` could not
re-establish a hold, `block_until_escape_is_safe` had no bounded exit, and the
operator's only escape was a second `Ctrl-C`.  That forced cancellation drops
the arm command, so the carton settled wherever it happened to be.

Two observations from the regression harness shaped the fix:

1. The release target was already re-based correctly.  `_make_cartesian_arm_target`
   builds `loaded = measured - axis * squeeze`, and the release step added
   `axis * (squeeze + spread)`, so the squeeze cancels exactly and the real
   commanded travel was always one spread per hand.  The 150 mm virtual squeeze
   never became real motion, and the singularity came from the descent alone.
2. A gap below the 50 mm clearance floor is withheld by `_start_gate_reason`
   rather than faulted, so the run stays recoverable if the operator
   repositions the pallet.

## Decision

Release the carton at the aligned pose by opening both hands along base Y only.

- **No vertical descent.** `planned_delta = min(gap * descent_fraction,
  maximum_planned_descent_m)` with `maximum_planned_descent_m = 0.0`.  The
  descent plan, the clearance floor, the freshness gates, and the
  `_lower_geometry_reached` check all stay wired, so restoring a descent later
  is a configuration change, not a code change.
- **Base `+/-Y` opening axis only.** `resolve_base_y_release_axis` snaps the
  measured inter-EEF axis to exactly `(0, +/-1, 0)`, taking the sign from the
  measured axis.  Base X and Z of the commanded target are bit-identical to the
  frozen plan, so release can never introduce vertical motion.
- **Bounded opening.** `release_spread_m = 0.030` per hand with a
  `maximum_release_spread_m = 0.040` ceiling, down from 0.120/0.120.
- **Axis guard.** If the measured axis deviates more than
  `release_axis_max_deviation_deg` (10 degrees) from base `+/-Y`, release fails
  closed with `release_axis_deviation` instead of shearing the carton sideways.
- **Release-height bound.** With no descent the carton falls the whole measured
  gap, so `maximum_release_gap_m` (0.120) rejects any plan whose gap would make
  the drop too tall.
- **Frozen-plan geometry.** The release target is built from
  `PlacementDescentPlan.right/left_target_base` instead of the live lowering
  target.  Both were equivalent in practice, but the plan is captured after
  arrival and wheel-stop and is bounded against the measured pose by
  `_validate_descent_plan_matches_state`, so the release target no longer
  depends on squeeze bookkeeping.
- **Bounded containment.** An expired or cancelled stream is classified as
  unrecoverable.  Containment keeps its unbounded wait while a hold is still
  recoverable, and after `unrecoverable_timeout_s` (30 s) it reports why no
  hold can be re-established instead of spinning.  Control faults now exit
  through their own CLI branch with a traceback and exit code 3 instead of an
  argparse usage error.

## Consequences

- `PlacementState.SEATED` and `seating_evidence` are misnomers under this
  decision.  There is no contact detection and no descent; `SEATED` means "the
  frozen residual-gap evidence still holds".  Renaming them would ripple into
  the `LOWERING_MODE` / `RELEASE_MODE` string contract shared with
  `pallet_runtime`, so the names stay and this ADR records the meaning.
- The carton is released from the full measured gap, at least 50 mm above the
  stack.  `maximum_release_gap_m` is the only control over that height and must
  be tightened against the measured `gap_m` during commissioning.
- Whether a 30 mm opening per hand reliably frees the carton depends on
  friction and carton deformation.  The 40 mm ceiling is the adjustment range.
- Physical re-verification is outstanding.  Software regression cannot prove
  singularity avoidance.

## Verification

- `Codex/tests/` was reintroduced (no tests were tracked in git before this
  change) with 66 hardware-free tests covering the placement sequencer, the
  descent-plan invariants, the release geometry, the release axis, the config
  schema, the runtime telemetry, containment escape, and the CLI error split.
- `ruff check`, `python -m compileall`, and
  `pallet.py replay` on `pallet_slot1` and `pallet_demo` all pass.
