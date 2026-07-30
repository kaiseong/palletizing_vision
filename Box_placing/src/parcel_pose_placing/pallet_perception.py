"""One frame in, one pallet observation out.

The control loop needs a single call here so that the flow it reads is drive,
arrive, place -- not depth scaling, hint assembly and freshness bookkeeping.  The
camera pose is an explicit input rather than a hidden lookup: live execution
derives it from measured torso/head forward kinematics every frame, replay uses
the configured audit transform, and a wrong camera pose is the failure mode most
easily mistaken for bad perception.

TODO(v2-recordings): ten sessions under recordings/box_* are
box-perception-recording-v2, which aligns depth onto the colour grid at 1280x720
while this pipeline works on the raw depth grid.  They cannot be replayed without
an adapter, and the resampled depth would not support dimension checks even then.
box_rotation in particular holds 244 frames of a rotating box that would show
where the yaw estimate becomes ambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class PalletFrameObservation:
    """What one frame produced, including the pose it was measured against."""

    scene: Any
    T_base_depth: np.ndarray
    held_proxy: Any
    estimator_finished_s: float
    decision_now_s: float
    result_fresh: bool
    scene_sample: dict[str, Any] | None
    accepted_scene_sequence: int


def observe_pallet_frame(
    frame: Any,
    *,
    root_config: Mapping[str, Any],
    contract: Any,
    estimator: Any,
    controller: Any | None,
    calibration_status: str,
    configured_T_base_depth: np.ndarray,
    configured_held_proxy: Any,
    capture_monotonic_s: float,
    accepted_scene_sequence: int,
    maximum_box_height_m: float,
    box_bottom_uncertainty_m: float,
) -> PalletFrameObservation:
    """Estimate one pallet observation from one captured frame.

    ``controller`` is ``None`` for perception-only runs, which then keep the
    configured camera pose and held-box proxy.  With a controller both are
    re-measured, because the depth camera rides on the torso/head chain.
    """

    from .pallet_runtime import (
        _controller_scene_sample,
        _fixed_ready_held_pose,
        _held_hint,
        _live_result_fresh,
        measured_T_base_from_depth,
    )

    T_base_depth = configured_T_base_depth
    held_proxy = configured_held_proxy
    if controller is not None:
        T_base_depth = measured_T_base_from_depth(
            root_config, controller.get_measured_T_base_head()
        )
        right_eef, left_eef = controller.get_measured_eef_transforms()
        held_proxy = _fixed_ready_held_pose(root_config, right_eef, left_eef)

    depth_m = frame.raw_depth_z16.astype(np.float32) * contract.depth_scale_m
    color = (
        frame.color_on_depth_bgr
        if frame.color_on_depth_bgr is not None
        else frame.raw_color_bgr
    )
    scene = estimator.estimate(
        depth_m,
        contract.depth_intrinsics,
        T_base_depth,
        timestamp_s=capture_monotonic_s,
        frame_id=frame.depth_frame_number,
        color_on_depth_bgr=color,
        held_box_hint=_held_hint(root_config, held_proxy),
        calibration_status=calibration_status,
    )
    estimator_finished_s = time.monotonic()
    decision_now_s = time.monotonic()
    result_fresh = _live_result_fresh(capture_monotonic_s, decision_now_s)

    sample: dict[str, Any] | None = None
    sequence = accepted_scene_sequence
    if result_fresh:
        sequence += 1
        sample = _controller_scene_sample(
            scene,
            held_proxy,
            frame_id=frame.depth_frame_number,
            accepted_observation_sequence=sequence,
            capture_timestamp_s=capture_monotonic_s,
            accepted_monotonic_s=decision_now_s,
            maximum_box_height_m=maximum_box_height_m,
            box_bottom_uncertainty_m=box_bottom_uncertainty_m,
        )
        stack_uncertainty = sample["stack_top_uncertainty_m"]
        if (
            sample["stack_top_z_base_m"] is None
            or stack_uncertainty is None
            or not math.isfinite(float(stack_uncertainty))
        ):
            # An unusable stack reading is not evidence; the interlock must not
            # see it as a contiguous frame.
            sample = None

    return PalletFrameObservation(
        scene=scene,
        T_base_depth=T_base_depth,
        held_proxy=held_proxy,
        estimator_finished_s=estimator_finished_s,
        decision_now_s=decision_now_s,
        result_fresh=result_fresh,
        scene_sample=sample,
        accepted_scene_sequence=sequence,
    )


__all__ = ["PalletFrameObservation", "observe_pallet_frame"]
