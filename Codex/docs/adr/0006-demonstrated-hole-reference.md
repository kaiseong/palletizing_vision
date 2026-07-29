# ADR 0006: Demonstrated hole reference and partial-edge acquisition

Status: accepted for supervised slot-1 hover commissioning

## Context

With a carton held at the fixed slot-1 ready posture, the `pallet_demo`
recording repeatedly shows clean metric near/front and image-right stack-edge
segments before the complete `148 x 149 mm` centre opening is visible. Those
segments are separated by the carried carton, so they do not prove a connected
metric L corner. The previous runtime nevertheless required the strict
connected-corner observation and therefore kept mobility at zero even while
both colored edge overlays were visible.

The `pallet_slot1` recording was captured at the operator-selected slot-1
arrival pose. Its 16 complete-hole dwell frames have the following robust
base-frame medians at the configured ready posture:

```text
hole center x = 0.865000 m
hole center y = 0.139523 m
line yaw      = -90.000 deg
```

The corresponding standard deviations are approximately `0.604 mm`,
`0.728 mm`, and `0.615 deg`. There is no independent ground truth, so this is
an operator-demonstrated nominal reference rather than an absolute pallet
calibration.

Recorded RealSense hardware frame numbers commonly advance by three or four
between processed observations. Requiring adjacent hardware counters for the
grip/clearance dwell makes the interlock impossible to satisfy even though the
accepted observations themselves are fresh and sequential.

A Jetson live run at commit `8af7223` exposed a second timing mismatch. The
processed capture cadence was approximately 10 Hz and a five-frame window
spanned a median `0.367 s`, but the controller required every historical frame
to remain within `0.20 s` of the current evaluation time. The five-frame gate
therefore could not complete: all 308 commands remained zero even though 303
frames had stable coarse edge evidence and conservative vertical clearance was
about `0.207 m`.

## Decision

- Keep strict `LCornerObservation.valid` unchanged. A disconnected edge pair
  has no metric corner position and must keep `corner_base=None`.
- Add a separate forward-acquisition edge-pair qualification. It may authorize
  only the existing bounded `+x`, stop-and-observe step. It cannot produce a
  stack centre, slot pose, lateral/yaw command, descent, contact, or release.
- Qualify the line-only path with independent support, residual, axis,
  orthogonality, gap-sanity, plane, branch, and five-frame stationary checks.
- Once the complete opening passes its stationary dwell, drive the currently
  observed hole centre/yaw to the demonstrated `pallet_slot1` base-frame
  reference. Use an explicit world-feature-to-body-reference measurement so
  the moving-base sign is not hidden behind held-box field names.
- Preserve the older geometric `slot1_target_base` as estimator/debug evidence;
  it is no longer the fine-servo destination for this fixed demonstrated pose.
- Track accepted control-observation sequence separately from the RealSense
  hardware frame counter. The dwell requires consecutive accepted fresh
  observations and strictly increasing source counters/timestamps, not
  numerically adjacent sensor counters.
- Separate current-frame freshness from historical clearance-dwell timing.
  Each scene must be fresh when accepted, the newest accepted scene must remain
  fresh, and five consecutive scenes must fit inside an explicit `0.50 s`
  evidence span. Keep the independent live actuation result-age cap at
  `0.15 s`; do not raise it or reduce the five-frame dwell to accommodate the
  measured 10 Hz processing cadence.
- Treat grip/vertical-clearance evidence as a rolling fresh interlock during an
  authorized coarse step and continuous fine alignment. Stationary evidence is
  separately mandatory for step authorization, post-stop reacquisition,
  complete-hole handoff, and arrival verification. Making clearance itself a
  stationary-only gate would cancel motion on its first moving frame.
- Show the selected dispatch result and motion-interlock reason in the live
  overlay and periodic console record. Visible geometry alone is not reported
  as proof that a command was transmitted.

## Consequences

The coarse phase can advance from the marker-docked starting region until the
centre opening becomes measurable, without pretending that occluded edge
segments recover a full stack pose. The fine phase directly reproduces the
operator-demonstrated camera/base relationship requested for slot 1. Accuracy
remains limited by the nominal camera registration, fixed ready/grasp posture,
carton deformation, and the lack of independent ground truth.

Mobile execution still requires the full explicit command set and all runtime
interlocks. `--ensure-slot1-ready` alone only verifies or restores posture and
does not enable the mobile base.
