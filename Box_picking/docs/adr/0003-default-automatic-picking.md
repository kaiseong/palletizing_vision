# ADR 0003: Default automatic box-picking entrypoint

## Status

Accepted, 2026-07-30.

Supersedes
[ADR 0002: Opt-in mobile alignment and seamless grasp handoff](0002-opt-in-mobile-auto-grab.md).

## Context

The box-picking phase is now treated as the commissioned default operation for
the `Box_picking` subsystem. Operators previously ran the same behavior through
`live_view.py --auto-grab --allow-nominal-registration`, but those flags made
the normal robot workflow look like a diagnostic opt-in path and left stale
`live-view` naming in the primary command surface.

The camera-to-base registration still uses the nominal unverified D435/RB-Y1
registration with the empirical `+0.050 m` base-y correction. That is acceptable
for the current workflow only when it remains visible to the operator and all
existing fail-closed gates stay active.

## Decision

1. Make `Box_picking/box_picking.py` the operator entrypoint. Running it with no
   arguments starts automatic box picking.
2. Enable the former `--auto-grab` behavior by default and remove
   `--auto-grab` from the public CLI.
3. Allow nominal registration by default and keep the explicit warning that base
   coordinates use `nominal_unverified` calibration with the empirical
   `+0.050 m` y correction.
4. Remove the `--allow-nominal-registration`, `live`, `live-view`, and
   `live_view.py` public command path from the box-picking operator workflow.
5. Keep fail-closed safety gates from the opt-in design: model/posture checks,
   calibrated torso/head verification or correction, mobile-ready arm command,
   bounded SE(2) stream, arrival hysteresis, zero latch, measured wheel stop,
   stream shutdown, and same-robot grasp handoff.
6. Retain expert output options for commissioning and review, including
   `--headless`, optional MP4 output, optional JSONL telemetry, config path, and
   robot-address overrides. These options must not be required for everyday
   operation and must not weaken the safety gates.

## Consequences

- The normal command is simply `python box_picking.py`.
- Headless operation remains `python box_picking.py --headless`.
- The command is no longer perception-only by default. It is an automatic robot
  execution entrypoint and must be launched only in an operator-supervised robot
  environment.
- The nominal camera registration warning remains mandatory because the
  empirical correction is not independent calibration.
- Historical ADR 0002 remains useful for understanding why the fail-closed
  motion handoff exists, but its opt-in CLI policy is no longer active.

## Verification

- CLI contract tests cover the default automatic execution path, absence of
  `--auto-grab` and `--allow-nominal-registration`, absence of `live` and
  `live-view` subcommands, retained `--headless`, and the nominal-registration
  warning.
- No physical robot validation was performed for this documentation decision.
