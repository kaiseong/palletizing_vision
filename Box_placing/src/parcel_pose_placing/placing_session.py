"""Stateful lower-service session for staged pallet placement.

The operator entrypoint owns the visible stage order.  This module keeps the
camera, rich scene gates, servo state, telemetry artifacts, and controller
preparation together while yielding at the two manipulation boundaries:
before the demonstrated place posture and after the sequencer authorizes the
demonstrated retreat posture.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any

from parcel_pose_common.realsense_adapter import RealSenseAdapter

from .pallet_perception import observe_pallet_frame
from .pallet_place import PlacementRequest
from . import pallet_runtime as _runtime


@dataclass(frozen=True, slots=True)
class PlacingAlignmentOutcome:
    """Result returned exactly at the alignment-to-place boundary."""

    descent_plan: Any | None
    handoff_ready: bool
    place_ready: bool
    user_cancelled: bool


@dataclass(frozen=True, slots=True)
class _ProcessedFrame:
    placement_output: Any | None
    decision: Any


@dataclass(frozen=True, slots=True)
class PerceivedPlacingFrame:
    """One captured frame after freshness validation and one facade call."""

    frame: Any
    observed: Any
    estimator_started_s: float
    validated_age_s: float
    source_s: float


class PlacingSession:
    """Own camera/per-frame services without owning entrypoint loop policy."""

    def __init__(
        self,
        *,
        controller: Any,
        controller_was_injected: bool,
        auto_place_slot1: bool,
        ensure_slot1_ready: bool,
        execute: bool,
        plan: Any,
        state: Any,
        stack: Any,
        headless: bool,
        log_jsonl: str | Path | None,
        max_frames: int | None,
        output_mp4: str | Path | None,
        robot_address: str,
        robot_power: str,
        root_config: Any,
        window_name: str,
    ) -> None:
        self.controller = controller
        self._controller_was_injected = bool(controller_was_injected)
        self._auto_place_slot1 = bool(auto_place_slot1)
        self._ensure_slot1_ready = bool(ensure_slot1_ready)
        self._execute = bool(execute)
        self._plan = plan
        self._stack = stack
        self._headless = bool(headless)
        self._log_jsonl = log_jsonl
        self._max_frames = max_frames
        self._output_mp4 = output_mp4
        self._robot_address = robot_address
        self._robot_power = robot_power
        self._root_config = root_config
        self._window_name = window_name

        self._T_base_depth = state.T_base_depth
        self._accepted_scene_sequence = state.accepted_scene_sequence
        self._box_bottom_uncertainty_m = state.box_bottom_uncertainty_m
        self._calibration_status = state.calibration_status
        self._containment = state.containment
        self._frame_count = state.frame_count
        self._frame_gate = state.frame_gate
        self._held_proxy = state.held_proxy
        self._last_placement_output = state.last_placement_output
        self._last_placement_runtime_diagnostics = (
            state.last_placement_runtime_diagnostics
        )
        self._maximum_box_height_m = state.maximum_box_height_m
        self._placement_alignment_ready_since_s = (
            state.placement_alignment_ready_since_s
        )
        self._placement_lowering_started = state.placement_lowering_started
        self._placement_release_started = state.placement_release_started
        self._scene_window = state.scene_window
        self._shutdown_pending = state.shutdown_pending

        self._resources = ExitStack()
        self._camera: Any | None = None
        self._camera_contract: Any | None = None
        self._log_stream: Any | None = None
        self._video_writer: Any | None = None
        self._window_created = False
        self._entered = False
        self._prepared = False
        self._prepare_attempted = False
        self._acquisition_open = False
        self._closed = False
        self._alignment_released = False
        self._pending_descent_plan: Any | None = None
        self._user_cancelled = False

    def __enter__(self) -> "PlacingSession":
        if self._entered or self._closed:
            raise RuntimeError("placing session cannot be entered twice")
        self._entered = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, traceback
        try:
            self.close()
        finally:
            if (
                exc is not None
                and self._execute
                and self._containment is not None
                and self._containment.robot_touched
            ):
                self._containment.confirm_persistent_support()
                self._containment.block_until_escape_is_safe()

    def prepare(self) -> None:
        """Connect and establish the existing loaded-ready zero-body stream."""

        if self._prepared:
            return
        if self._prepare_attempted:
            raise RuntimeError("placing ready preparation was already attempted")
        self._prepare_attempted = True
        if not self._execute:
            self._prepared = True
            return
        if self.controller is None:
            raise RuntimeError("live placing session has no authorized controller")

        if self._ensure_slot1_ready and not self._controller_was_injected:
            from .pallet_ready import ensure_slot1_ready_from_config

            ensure_slot1_ready_from_config(
                self._root_config,
                address=self._robot_address,
                power=self._robot_power,
                slot=self._plan.selected_slot,
                prepare_for_stream=True,
            )
            self._calibration_status = (
                "nominal_unverified_ready_posture_checked_at_start"
            )

        self._containment = _runtime.ActuationContainmentState(self.controller)
        _runtime._prepare_loaded_ready_actuation(
            self.controller,
            float(
                _runtime._section(self._root_config, "robot").get(
                    "ready_transition_minimum_time_s", 5.0
                )
            ),
            self._containment,
        )
        self._T_base_depth = _runtime.measured_T_base_from_depth(
            self._root_config,
            self.controller.get_measured_T_base_head(),
        )
        right_eef, left_eef = self.controller.get_measured_eef_transforms()
        self._held_proxy = _runtime._fixed_ready_held_pose(
            self._root_config,
            right_eef,
            left_eef,
        )
        self._calibration_status = "nominal_unverified_operator_accepted"
        self._prepared = True
        print(
            "warning: loaded slot-1 MVP uses fixed-ready FK/EEF geometry and "
            "depth clearance evidence; placement does not read F/T",
            file=sys.stderr,
            flush=True,
        )

    @property
    def user_cancelled(self) -> bool:
        """Expose cancellation state so the entrypoint owns loop exit."""

        return self._user_cancelled

    def has_frame_budget(self) -> bool:
        """Return whether the entrypoint may acquire another frame."""

        return self._max_frames is None or self._frame_count < self._max_frames

    def open_acquisition(self) -> None:
        """Open camera/telemetry resources without consuming a frame."""

        if self._execute and not self._prepared:
            raise RuntimeError("placement lifecycle must start before acquisition")
        self._ensure_acquisition_open()

    def acquire_frame(self) -> Any:
        """Capture exactly one frame; the entrypoint chooses loop continuation."""

        self._ensure_acquisition_open()
        assert self._camera is not None
        return self._camera.capture()

    def handle_interrupt(self) -> None:
        """Apply containment for one interrupt without hiding the exit branch."""

        self._user_cancelled = True
        if not self._execute:
            return
        assert self._containment is not None
        if self._containment.request_shutdown_hold(
            next_owner="operator-successor-required"
        ):
            return
        self._stack.authority.request_shutdown_hold()
        self._shutdown_pending = True

    def accept_descent_plan(self, plan: Any) -> None:
        """Record the entrypoint-validated alignment handoff plan."""

        if plan is None or not bool(getattr(plan, "valid", False)):
            raise RuntimeError("placement sequencer returned no valid descent plan")
        self._pending_descent_plan = plan

    def release_alignment(self) -> bool:
        """Revoke nonzero base authority while retaining exact-zero body support."""

        if self._alignment_released:
            return True
        if self._pending_descent_plan is None:
            raise RuntimeError("alignment cannot be released before a place plan")
        self._stack.authority.release_alignment_for_placement()
        self._alignment_released = True
        return True

    def begin_release_observation(self) -> None:
        """Enter post-place observation after stopped alignment is proven."""

        if not self._alignment_released or self._pending_descent_plan is None:
            raise RuntimeError("release authorization requires stopped alignment")
        self._placement_lowering_started = True

    def align(self, lifecycle: Any | None) -> PlacingAlignmentOutcome:
        """Legacy wrapper assembled from loop-free primitives.

        Public entrypoints must own this loop directly; this remains only for
        callers outside the operator entrypoint during migration.
        """

        if self._execute and lifecycle is None:
            raise RuntimeError("live alignment requires the placement lifecycle")
        self.open_acquisition()
        while self.has_frame_budget():
            try:
                frame = self.acquire_frame()
            except KeyboardInterrupt:
                self.handle_interrupt()
                break
            perceived = self.perceive_frame(frame)
            base_motion = self.decide_base_motion(perceived)
            placement_step = self.advance_placement(perceived, base_motion)
            self.record_frame(perceived, base_motion, placement_step)
            self.finish_frame()
            placement = placement_step.placement_output
            if self._user_cancelled:
                break
            if placement is not None and placement.faulted:
                raise RuntimeError(
                    f"slot-1 placement sequencer faulted: {placement.reason}"
                )
            if (
                placement is not None
                and placement.request is PlacementRequest.LOWER_CARTESIAN_PLANNED
            ):
                self.accept_descent_plan(placement.descent_plan)
                return PlacingAlignmentOutcome(
                    descent_plan=placement.descent_plan,
                    handoff_ready=True,
                    place_ready=True,
                    user_cancelled=False,
                )

        return PlacingAlignmentOutcome(
            descent_plan=None,
            handoff_ready=False,
            place_ready=False,
            user_cancelled=self._user_cancelled,
        )

    def await_release_authorization(self) -> bool:
        """Legacy wrapper for entrypoint-owned post-place frame stepping."""

        self.begin_release_observation()
        while self.has_frame_budget():
            try:
                frame = self.acquire_frame()
            except KeyboardInterrupt:
                self.handle_interrupt()
                return False
            perceived = self.perceive_frame(frame)
            base_motion = self.decide_base_motion(perceived)
            placement_step = self.advance_placement(perceived, base_motion)
            self.record_frame(perceived, base_motion, placement_step)
            self.finish_frame()
            placement = placement_step.placement_output
            if self._user_cancelled:
                return False
            if placement is None:
                continue
            if placement.faulted:
                raise RuntimeError(
                    f"slot-1 release authorization faulted: {placement.reason}"
                )
            if (
                placement.request is PlacementRequest.SPREAD_RELEASE
                and placement.release_authorized
            ):
                return True
        return False

    def close(self) -> None:
        """Close camera, display, video, and telemetry exactly once."""

        if self._closed:
            return
        self._closed = True
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
        if self._log_stream is not None:
            self._log_stream.close()
            self._log_stream = None
        if self._window_created:
            try:
                import cv2  # type: ignore[import-not-found]

                cv2.destroyWindow(self._window_name)
            except Exception:
                pass
            self._window_created = False
        self._resources.close()
        self._camera = None
        self._camera_contract = None

    def _frame_limit_reached(self) -> bool:
        return self._max_frames is not None and self._frame_count >= self._max_frames

    def _ensure_acquisition_open(self) -> None:
        if self._acquisition_open:
            return
        if not self._entered or self._closed:
            raise RuntimeError("placing session is not active")
        self._log_stream = _runtime._open_log(
            None if self._log_jsonl is None else Path(self._log_jsonl)
        )
        self._camera = self._resources.enter_context(
            RealSenseAdapter(self._plan.stream_config)
        )
        self._camera_contract = _runtime.validate_live_camera_profile(
            self._camera.active_profile_metadata(),
            self._root_config,
        )
        if not self._headless:
            import cv2  # type: ignore[import-not-found]

            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
            self._window_created = True
        self._acquisition_open = True

    def perceive_frame(self, frame: Any) -> PerceivedPlacingFrame:
        """Validate and perceive exactly one already-acquired frame."""

        assert self._camera_contract is not None
        received_s = time.monotonic()
        estimator_started_s = received_s
        validated_age_s = self._frame_gate.validate(frame)
        source_s = received_s - validated_age_s
        observed = observe_pallet_frame(
            frame,
            slot=self._plan.selected_slot,
            root_config=self._root_config,
            contract=self._camera_contract,
            estimator=self._stack.estimator,
            controller=self.controller if self._execute else None,
            calibration_status=self._calibration_status,
            configured_T_base_depth=self._T_base_depth,
            configured_held_proxy=self._held_proxy,
            capture_monotonic_s=source_s,
            accepted_scene_sequence=self._accepted_scene_sequence,
            maximum_box_height_m=self._maximum_box_height_m,
            box_bottom_uncertainty_m=self._box_bottom_uncertainty_m,
        )
        self._T_base_depth = observed.T_base_depth
        self._held_proxy = observed.held_proxy
        self._accepted_scene_sequence = observed.accepted_scene_sequence
        _runtime._update_scene_window(
            self._scene_window,
            frame_result_fresh=observed.result_fresh,
            fresh_sample=observed.scene_sample,
        )
        return PerceivedPlacingFrame(
            frame=frame,
            observed=observed,
            estimator_started_s=estimator_started_s,
            validated_age_s=validated_age_s,
            source_s=source_s,
        )

    def decide_base_motion(self, perceived: PerceivedPlacingFrame) -> Any:
        """Run exactly one x/y/yaw base decision for a perceived frame."""

        observed = perceived.observed
        return _runtime._decide_base_motion(
            acquisition_config=self._stack.acquisition_config,
            acquisition_servo=self._stack.acquisition_servo,
            authority=self._stack.authority,
            auto_place_slot1=self._auto_place_slot1,
            controller=self.controller,
            decision_now_s=observed.decision_now_s,
            estimator_config=self._stack.estimator_config,
            execute=self._execute,
            frame_result_fresh=observed.result_fresh,
            frame_source_monotonic_s=perceived.source_s,
            geometry_only_policy_enabled=self._plan.geometry_only_policy_enabled,
            hole_gate=self._stack.hole_gate,
            l_corner_gate=self._stack.l_corner_gate,
            placement_lowering_started=self._placement_lowering_started,
            placement_release_started=self._placement_release_started,
            placement_sequencer=self._stack.placement_sequencer,
            pose_result=observed.pose_result,
            scene=observed.scene,
            scene_window=self._scene_window,
            servo=self._stack.servo,
            servo_bridge=self._stack.servo_bridge,
            shutdown_pending=self._shutdown_pending,
            slot1_hole_reference=self._stack.slot1_hole_reference,
        )

    def advance_placement(
        self,
        perceived: PerceivedPlacingFrame,
        base_motion: Any,
    ) -> Any:
        """Advance safety/sequencer state once without dispatching manipulation."""

        observed = perceived.observed
        placement_step = _runtime._advance_placement(
            authority=self._stack.authority,
            auto_place_slot1=self._auto_place_slot1,
            containment=self._containment,
            controller=self.controller,
            decision=base_motion.decision,
            decision_now_s=observed.decision_now_s,
            decision_owner=base_motion.decision_owner,
            decision_source_max_age_s=base_motion.decision_source_max_age_s,
            decision_source_timestamp_s=base_motion.decision_source_timestamp_s,
            estimator_config=self._stack.estimator_config,
            execute=self._execute,
            frame=perceived.frame,
            frame_source_monotonic_s=perceived.source_s,
            motion_interlocks_ok=base_motion.motion_interlocks_ok,
            placement_config=self._stack.placement_config,
            placement_motion_active=base_motion.placement_motion_active,
            placement_sequencer=self._stack.placement_sequencer,
            root_config=self._root_config,
            scene=observed.scene,
            stationary=base_motion.stationary,
            vision_release_policy_enabled=self._plan.vision_release_policy_enabled,
            last_placement_output=self._last_placement_output,
            last_placement_runtime_diagnostics=(
                self._last_placement_runtime_diagnostics
            ),
            placement_alignment_ready_since_s=(
                self._placement_alignment_ready_since_s
            ),
            placement_lowering_started=self._placement_lowering_started,
            placement_release_started=self._placement_release_started,
            dispatch_manipulation=False,
        )
        self._last_placement_output = placement_step.last_placement_output
        self._last_placement_runtime_diagnostics = (
            placement_step.last_placement_runtime_diagnostics
        )
        self._placement_alignment_ready_since_s = (
            placement_step.placement_alignment_ready_since_s
        )
        self._placement_lowering_started = placement_step.placement_lowering_started
        self._placement_release_started = placement_step.placement_release_started
        return placement_step

    def finish_frame(self) -> None:
        """Commit one fully recorded frame to the session budget."""

        self._frame_count += 1

    def _process_next_frame(self) -> _ProcessedFrame | None:
        """Legacy one-frame wrapper built from the public primitives."""

        try:
            frame = self.acquire_frame()
        except KeyboardInterrupt:
            self.handle_interrupt()
            return None
        perceived = self.perceive_frame(frame)
        base_motion = self.decide_base_motion(perceived)
        placement_step = self.advance_placement(perceived, base_motion)
        self.record_frame(perceived, base_motion, placement_step)
        self.finish_frame()
        if self._user_cancelled:
            return None
        return _ProcessedFrame(
            placement_output=placement_step.placement_output,
            decision=placement_step.decision,
        )

    def record_frame(
        self,
        perceived: PerceivedPlacingFrame,
        base_motion: Any,
        placement_step: Any,
    ) -> None:
        """Keep overlay/telemetry as observability, never loop policy."""

        frame = perceived.frame
        observed = perceived.observed
        estimator_started_s = perceived.estimator_started_s
        validated_age_s = perceived.validated_age_s
        color = (
            frame.color_on_depth_bgr
            if frame.color_on_depth_bgr is not None
            else frame.raw_color_bgr
        )
        placement = placement_step.placement_output or self._last_placement_output
        placement_runtime = (
            placement_step.placement_runtime_diagnostics
            or self._last_placement_runtime_diagnostics
        )
        overlay = _runtime.draw_live_overlay(
            color,
            observed.scene,
            self._stack.estimator.last_evidence,
            self._held_proxy,
            placement_step.decision,
            self._T_base_depth,
            self._camera_contract.depth_intrinsics,
            self._stack.slot1_hole_reference,
            self._stack.estimator_config.geometry,
            execute=self._execute,
            acquisition=base_motion.acquisition_output,
            l_gate=base_motion.l_status,
            hole_gate=base_motion.hole_status,
            stationary_source=base_motion.stationary_source,
            motion_interlock_reason=base_motion.motion_interlock_reason,
            dispatch_result=placement_step.dispatch_result,
            placement=placement,
            placement_runtime=placement_runtime,
        )
        if self._video_writer is None and self._output_mp4 is not None:
            self._video_writer = _runtime._open_video(
                Path(self._output_mp4),
                overlay.shape[:2],
                self._plan.fps,
            )
        if self._video_writer is not None:
            self._video_writer.write(overlay)

        finished_s = time.monotonic()
        loop_timing = {
            "capture_age_s": float(validated_age_s),
            "estimator_ms": 1000.0
            * (observed.estimator_finished_s - estimator_started_s),
            "post_estimator_decision_ms": 1000.0
            * (finished_s - observed.estimator_finished_s),
            "loop_ms": 1000.0 * (finished_s - estimator_started_s),
        }
        record = _runtime._telemetry_record(
            self._frame_count,
            float(frame.hardware_timestamp_ms or frame.depth_timestamp_ms),
            observed.scene,
            self._held_proxy,
            placement_step.decision,
            execute=self._execute,
            controller=self.controller,
            acquisition=base_motion.acquisition_output,
            l_gate=base_motion.l_status,
            hole_gate=base_motion.hole_status,
            stationary_source=base_motion.stationary_source,
            odometry=base_motion.odometry,
            odometry_error=base_motion.odometry_error,
            motion_interlocks_ok=base_motion.motion_interlocks_ok,
            motion_interlock_reason=base_motion.motion_interlock_reason,
            grip_result=base_motion.grip_result,
            dispatch_result=placement_step.dispatch_result,
            T_base_depth=self._T_base_depth,
            slot1_hole_reference=self._stack.slot1_hole_reference,
            placement=placement,
            placement_runtime_diagnostics=placement_runtime,
            loop_timing=loop_timing,
            geometry=self._stack.estimator_config.geometry,
            estimator_config=self._stack.estimator_config,
            bridge_diagnostics=base_motion.bridge_diagnostics,
        )
        _runtime._write_record(self._log_stream, record)

        if not self._headless:
            import cv2  # type: ignore[import-not-found]

            cv2.imshow(self._window_name, overlay)
            key = int(cv2.waitKey(1)) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                self._user_cancelled = True


def open_placing_session(
    *,
    controller: Any | None,
    auto_place_slot1: bool,
    ensure_slot1_ready: bool,
    execute: bool,
    plan: Any,
    state: Any,
    stack: Any,
    headless: bool,
    log_jsonl: str | Path | None,
    max_frames: int | None,
    output_mp4: str | Path | None,
    robot_address: str,
    robot_power: str,
    root_config: Any,
    window_name: str,
) -> PlacingSession:
    """Build lower collaborators after authority and pure plan validation."""

    injected = controller is not None
    if execute and controller is None:
        from .pallet_control import PalletControlConfig, RBY1PalletController

        controller = RBY1PalletController(
            execute=True,
            config=PalletControlConfig.from_root_config(
                root_config,
                address_override=robot_address,
                slot=plan.selected_slot,
            ),
        )
    return PlacingSession(
        controller=controller,
        controller_was_injected=injected,
        auto_place_slot1=auto_place_slot1,
        ensure_slot1_ready=ensure_slot1_ready,
        execute=execute,
        plan=plan,
        state=state,
        stack=stack,
        headless=headless,
        log_jsonl=log_jsonl,
        max_frames=max_frames,
        output_mp4=output_mp4,
        robot_address=robot_address,
        robot_power=robot_power,
        root_config=root_config,
        window_name=window_name,
    )


__all__ = [
    "PerceivedPlacingFrame",
    "PlacingAlignmentOutcome",
    "PlacingSession",
    "open_placing_session",
]
