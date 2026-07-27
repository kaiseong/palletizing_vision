# ADR 0001: Metric top-plane parcel estimator

## Status

Accepted, 2026-07-27.

## Context

RB-Y1 observes one upright, closed cardboard parcel on a fixed-height table with a rigid D435. The parcel dimensions are fixed at 400 x 250 x 150 mm. The center remains in the image, but physical edges may be cropped. Available sensor evidence is raw RGB, raw depth, intrinsics, and factory stream extrinsics.

The supplied `link_head_2` to RGB-centered pose is nominal rather than calibrated, and unlabeled arbitrary box placements cannot identify the global base-plane camera X/Y/yaw transform. The perception output must therefore separate geometric observability from base registration validity.

## Decision

Use native raw-depth geometry as the canonical path:

1. Fit and persist an empty-table plane in the D435 depth optical frame.
2. Offset that plane by 150 mm along its oriented normal to define the parcel top plane.
3. Use raw depth only to select a top-plane slab and candidate component.
4. Intersect selected depth-pixel rays with the exact top plane.
5. Fit a fixed 400 x 250 mm rectangle in continuous metric plane coordinates, optimizing center and line yaw only.
6. Evaluate long/short side-swap candidates, crop-censored edge support, fit conditioning, and best-versus-second margin.
7. Return null or a feasible set for underconstrained fields.
8. Transform to RB-Y1 base only when the complete target-from-source transform chain is supplied; keep calibration state explicit.

RGB is optional support for candidate rejection or boundary weighting after association with the raw-depth grid. It is never required for metric scale and does not replace plane geometry.

## Alternatives considered

- Image-space bounding boxes, Hough angle, or `minAreaRect` as the final estimator: rejected because perspective, cropping, and depth-dependent pixel scale bias the center and yaw.
- Per-frame 3D PCA/minimum rectangle on noisy depth: rejected as a final estimator because depth holes and partial crop shrink or rotate the observed support.
- Copy the existing D405 yellow/open-rim picking pipeline: rejected because its camera, object, segmentation, dimensions, coordinate output, and downstream control assumptions do not match this task.
- Import that legacy package in place: rejected because it creates cross-project coupling and exposes robot-command-adjacent code.

## Consequences

- The estimator is deterministic, metric, replayable, and explicit about physically missing information.
- A table calibration recording is required before meaningful box estimation.
- Cropped or ambiguous frames are rejected more often instead of receiving optimistic point poses.
- Absolute base accuracy cannot be claimed until an independent base-referenced calibration/label source exists.
- Live D435 behavior must be smoke-tested on Jetson because the development environment does not provide `pyrealsense2` or camera hardware.

## Forward constraints

- Do not add the D435 front-glass `-4.2 mm` value to Z16 or deprojected points.
- Do not deproject raw depth using color intrinsics.
- Do not optimize base-plane camera X/Y/yaw from unlabeled arbitrary box recordings.
- Do not emit robot motion, grasp, power, contact, trajectory, or end-effector command fields from this package.

