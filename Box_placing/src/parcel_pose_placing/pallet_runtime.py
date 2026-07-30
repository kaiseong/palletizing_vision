"""Live D435 facade and in-process slot-1 coordinator.

Plain ``pallet live`` is perception-only: it never imports the RB-Y1 SDK and
never connects to a robot.  Explicit loaded slot-1 execution first verifies the
configured ready posture, then starts one combined owner whose every packet
contains torso/head hold, both-arm impedance hold, and SE(2) mobility.  This
MVP starts only after the previous process has ended and the current session
has freshly measured the already-held ready posture.

The default execute path terminates alignment in a persistent zero-mobility
body hold.  Slot-1 lowering/release is available only behind explicit runtime
and config commissioning gates, and remains exact-zero mobility throughout.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping, Protocol, TextIO

import numpy as np

from parcel_pose_common.angles import normalize_line_angle_rad
from parcel_pose_common.models import CameraIntrinsics
from parcel_pose_common.mobile_servo import VelocityCommand
from .pallet_acquisition import (
    AcquisitionConfig,
    AcquisitionDecision,
    AcquisitionOutput,
    AcquisitionState,
    CoarseFineAuthority,
    ForwardAcquireServo,
    HoleGateStatus,
    LCornerGateStatus,
    OdometrySample,
    PalletControlOwner,
    StationaryHoleGate,
    StationaryLCornerGate,
)
from .pallet_models import (
    HeldBoxHint,
    PalletGeometry,
    PalletSceneObservation,
    Slot1HoleReference,
    load_slot1_hole_reference,
    require_slot_member,
)
from .pallet_control import MobilityCommand
from .pallet_place import (
    PlacementConfig,
    PlacementDescentPlan,
    PlacementInput,
    PlacementOutput,
    PlacementRequest,
    PlacementState,
    Slot1PlacementSequencer,
)
from .pallet_perception import observe_pallet_frame
from .pallet_telemetry import (
    _placement_telemetry_payload,
    _servo_measurement_source,
    _telemetry_record,
)
from .pallet_visualization import (
    draw_live_overlay,
)
from .pallet_servo import (
    PalletServoConfig,
    PalletServoObservation,
    PalletServoOutput,
    PalletServoState,
    PalletSlot1Servo,
    WheelMotionMeasurement,
)
from parcel_pose_common.realsense_adapter import D435StreamConfig, RealSenseAdapter
from parcel_pose_common.transforms import compose_base_from_depth


_MAX_LIVE_CONTROL_RESULT_AGE_S = 0.15
_MAX_ODOMETRY_PREDICTION_AGE_S = 0.30


class _ControllerLike(Protocol):
    """Narrow boundary implemented by ``RBY1PalletController``."""

    owner_epoch: str
    is_connected: bool

    def connect(self) -> None: ...

    def bootstrap_loaded_slot1_ready(
        self,
        *,
        loaded_box_acknowledged: bool,
    ) -> Any: ...

    def send_ready_transition_once(
        self,
        minimum_time_s: float = 5.0,
        *,
        on_send_attempt: Callable[[], None] | None = None,
    ) -> Any: ...

    def wait_ready_transition_ack(self, command_id: Any) -> Any: ...

    def start_combined_stream(self) -> None: ...

    def ensure_persistent_zero_body_hold(self) -> None: ...

    def get_measured_T_base_head(self) -> np.ndarray: ...

    def get_measured_eef_transforms(self) -> tuple[np.ndarray, np.ndarray]: ...

    def get_measured_state(self) -> Any: ...

    def placement_telemetry(self) -> Any: ...

    def start_cartesian_lowering_hold(
        self,
        *,
        descent_plan: PlacementDescentPlan,
    ) -> Any: ...

    def start_cartesian_release_hold(
        self,
        *,
        release_spread_m: float | None = None,
    ) -> Any: ...

    def fail_closed_cartesian_placement_hold(
        self,
        *,
        reason: str,
    ) -> Any: ...

    def get_measured_odometry(self) -> tuple[np.ndarray, int, float]: ...

    def evaluate_grip_and_clearance_dwell(
        self,
        scene_window: list[Any],
        *,
        allow_fixed_ready_geometry_only: bool = False,
    ) -> Any: ...

    def wheel_stop_status(self) -> Any: ...

    def wait_for_wheel_stop(self, timeout_s: float) -> Any: ...

    def reverify_wheel_stop_after_stream_start(self, timeout_s: float) -> Any: ...

    def send_cycle(self, mobility: Any, *, owner_epoch: str | None = None) -> int: ...

    def send_zero_mobility_hold(self, *, latch: bool = True) -> None: ...

    def resume_mobility(self, *, owner_epoch: str) -> None: ...

    def transfer_owner(self, next_owner: str) -> Any: ...

    def close(self, *, force: bool = False) -> bool: ...

    def telemetry(self) -> Any: ...


@dataclass(frozen=True, slots=True)
class LiveCameraContract:
    camera_name: str
    camera_serial: str
    depth_scale_m: float
    depth_intrinsics: CameraIntrinsics
    color_intrinsics: CameraIntrinsics


@dataclass(frozen=True, slots=True)
class HeldPoseProxy:
    center_base_xyz_m: tuple[float, float, float]
    yaw_base_rad: float
    source: str


@dataclass(slots=True)
class ServoObservationBridge:
    """Propagate one locked metric observation through a short visual dropout.

    The visual anchor is never advanced by predicted samples, so repeated
    odometry updates cannot extend the 0.30 s authority window.  Fresh visual
    geometry must be accepted by the fine controller before it can seed or
    refresh the bridge.
    """

    maximum_prediction_age_s: float = _MAX_ODOMETRY_PREDICTION_AGE_S
    odometry_freshness_s: float = 0.20
    _locked: bool = False
    _observation: PalletServoObservation | None = None
    _odometry: OdometrySample | None = None
    _visual_timestamp_s: float | None = None

    def __post_init__(self) -> None:
        for name in ("maximum_prediction_age_s", "odometry_freshness_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            setattr(self, name, value)
        if self.maximum_prediction_age_s > _MAX_ODOMETRY_PREDICTION_AGE_S + 1e-12:
            raise ValueError("visual dropout prediction cannot exceed 0.30 s")

    def resolve(
        self,
        observation: PalletServoObservation,
        odometry: OdometrySample | None,
        *,
        now_s: float,
    ) -> tuple[PalletServoObservation, dict[str, Any]]:
        """Return fresh geometry or a bounded odometry-propagated observation."""

        now = float(now_s)
        diagnostics: dict[str, Any] = {
            "odometry_prediction_used": False,
            "dropout_age_s": None,
            "prediction_ttl_s": self.maximum_prediction_age_s,
            "prediction_reason": "fresh_visual_observation",
            "visual_anchor_timestamp_s": self._visual_timestamp_s,
            "odometry_timestamp_s": None if odometry is None else odometry.timestamp_s,
            "odometry_sequence": (
                None if odometry is None else getattr(odometry, "sequence", None)
            ),
        }
        if observation.valid:
            return observation, diagnostics

        if not self._locked or self._observation is None or self._odometry is None:
            diagnostics["prediction_reason"] = "initial_metric_lock_missing"
            return observation, diagnostics
        if self._visual_timestamp_s is None:
            diagnostics["prediction_reason"] = "visual_anchor_timestamp_missing"
            return observation, diagnostics

        dropout_age_s = now - self._visual_timestamp_s
        diagnostics["dropout_age_s"] = dropout_age_s
        if (
            not math.isfinite(dropout_age_s)
            or dropout_age_s < 0.0
            or dropout_age_s > self.maximum_prediction_age_s + 1e-12
        ):
            diagnostics["prediction_reason"] = "visual_dropout_ttl_expired"
            return PalletServoObservation.invalid(
                observation.timestamp_s,
                "visual_dropout_ttl_expired",
                reference_source=observation.reference_source,
            ), diagnostics
        if self._conflicting_invalid_reasons(observation.rejection_reasons):
            diagnostics["prediction_reason"] = "conflicting_metric_geometry"
            return observation, diagnostics
        if not self._odometry_is_fresh(odometry, now):
            diagnostics["prediction_reason"] = "odometry_stale_or_invalid"
            return PalletServoObservation.invalid(
                observation.timestamp_s,
                "odometry_stale_or_invalid",
                reference_source=observation.reference_source,
            ), diagnostics

        assert odometry is not None
        anchor = self._observation
        assert anchor.current_observed_feature_center_base is not None
        assert anchor.current_observed_feature_yaw_base_rad is not None
        assert anchor.demonstrated_body_reference_center_base is not None
        assert anchor.demonstrated_body_reference_yaw_base_rad is not None
        assert anchor.axis_branch is not None

        anchor_xy = np.asarray(
            anchor.current_observed_feature_center_base,
            dtype=np.float64,
        )
        c0 = math.cos(self._odometry.yaw_rad)
        s0 = math.sin(self._odometry.yaw_rad)
        R_odom_from_anchor = np.asarray(((c0, -s0), (s0, c0)))
        point_odom = (
            R_odom_from_anchor @ anchor_xy
            + np.asarray((self._odometry.x_m, self._odometry.y_m))
        )
        c1 = math.cos(odometry.yaw_rad)
        s1 = math.sin(odometry.yaw_rad)
        R_odom_from_current = np.asarray(((c1, -s1), (s1, c1)))
        point_current = R_odom_from_current.T @ (
            point_odom - np.asarray((odometry.x_m, odometry.y_m))
        )
        yaw_current = normalize_line_angle_rad(
            anchor.current_observed_feature_yaw_base_rad
            + self._odometry.yaw_rad
            - odometry.yaw_rad
        )
        propagated = PalletServoObservation(
            timestamp_s=float(odometry.timestamp_s),
            current_observed_feature_center_base=(
                float(point_current[0]),
                float(point_current[1]),
            ),
            current_observed_feature_yaw_base_rad=yaw_current,
            demonstrated_body_reference_center_base=(
                float(anchor.demonstrated_body_reference_center_base[0]),
                float(anchor.demonstrated_body_reference_center_base[1]),
            ),
            demonstrated_body_reference_yaw_base_rad=float(
                anchor.demonstrated_body_reference_yaw_base_rad
            ),
            axis_branch=anchor.axis_branch,
            reference_source=anchor.reference_source,
            odometry_propagated=True,
            propagation_age_s=dropout_age_s,
        )
        diagnostics.update(
            {
                "odometry_prediction_used": True,
                "prediction_reason": "bounded_odometry_prediction",
                "visual_anchor_timestamp_s": self._visual_timestamp_s,
            }
        )
        return propagated, diagnostics

    def commit(
        self,
        observation: PalletServoObservation,
        odometry: OdometrySample | None,
        output: PalletServoOutput,
        *,
        now_s: float,
    ) -> None:
        """Refresh the bridge only from a fresh controller-accepted sample."""

        if (
            not observation.valid
            or not output.measurement_accepted
            or output.state
            not in {
                PalletServoState.TRACKING,
                PalletServoState.ARRIVAL_EVIDENCE,
                PalletServoState.ARRIVAL_WHEEL_STOP,
                PalletServoState.ARRIVED_HOLD,
            }
            or not self._odometry_is_fresh(odometry, now_s)
        ):
            return
        assert odometry is not None
        self._observation = observation
        self._odometry = odometry
        self._visual_timestamp_s = float(observation.timestamp_s)
        self._locked = True

    def clear(self) -> None:
        self._locked = False
        self._observation = None
        self._odometry = None
        self._visual_timestamp_s = None

    def _odometry_is_fresh(
        self,
        odometry: OdometrySample | None,
        now_s: float,
    ) -> bool:
        if odometry is None:
            return False
        values = (
            odometry.timestamp_s,
            odometry.x_m,
            odometry.y_m,
            odometry.yaw_rad,
        )
        age_s = float(now_s) - float(odometry.timestamp_s)
        return bool(
            all(math.isfinite(float(value)) for value in values)
            and -1e-9 <= age_s <= self.odometry_freshness_s + 1e-12
        )

    @staticmethod
    def _conflicting_invalid_reasons(reasons: tuple[str, ...]) -> bool:
        conflict_tokens = (
            "axis",
            "branch",
            "reflected",
            "opposite",
            "mismatch",
            "jump",
            "transform",
            "calibration",
            "nonhorizontal",
            "normal",
        )
        return any(
            token in str(reason).lower()
            for reason in reasons
            for token in conflict_tokens
        )


@dataclass(slots=True)
class ActuationContainmentState:
    """Track whether an actuating runtime may safely unwind.

    Once destination commands may exist, escape is permitted only after an
    acknowledged zero-mobility body hold (or the explicit second-interrupt
    forced-cancel path).
    """

    controller: _ControllerLike
    robot_touched: bool = False
    destination_commanded: bool = False
    destination_steady: bool = False
    persistent_support_confirmed: bool = False
    successor_acknowledged: bool = False
    handoff_offer_created: bool = False
    forced_cancel: bool = False
    interrupt_count: int = 0
    support_owner: str = "standalone_ready_session"
    last_hold_error: str | None = None
    # An expired/cancelled command stream can never be recovered into a zero
    # body hold, so waiting forever only delays the operator's decision.  A
    # recoverable hold keeps the original unbounded contract.
    unrecoverable_timeout_s: float = 30.0
    hold_unrecoverable: bool = False
    _unrecoverable_since_s: float | None = None
    _clock: Callable[[], float] = time.monotonic

    def mark_robot_touch(self) -> None:
        self.robot_touched = True

    def mark_destination_commanded(self) -> None:
        self.destination_commanded = True
        self.support_owner = "destination_transition"

    def mark_destination_steady(self) -> None:
        self.destination_steady = True
        self.support_owner = "destination_steady"

    @staticmethod
    def _hold_is_unrecoverable(detail: str) -> bool:
        """Detect a stream the SDK will refuse for the rest of this session."""

        text = detail.lower()
        return any(
            fragment in text
            for fragment in (
                "stream is expired",
                "stream expired",
                "stream is closed",
                "stream was cancelled",
                "stream was canceled",
            )
        )

    def _note_hold_failure(self, detail: str) -> None:
        self.last_hold_error = detail
        if not self._hold_is_unrecoverable(detail):
            return
        self.hold_unrecoverable = True
        if self._unrecoverable_since_s is None:
            self._unrecoverable_since_s = self._clock()

    def unrecoverable_elapsed_s(self) -> float:
        if self._unrecoverable_since_s is None:
            return 0.0
        return max(0.0, self._clock() - self._unrecoverable_since_s)

    def escape_deadline_reached(self) -> bool:
        return bool(
            self.hold_unrecoverable
            and self.unrecoverable_elapsed_s() >= self.unrecoverable_timeout_s
        )

    @staticmethod
    def _telemetry_confirms_zero_body_hold(telemetry: Any) -> bool:
        mobility = getattr(telemetry, "last_sent_mobility", None)
        phase = getattr(getattr(telemetry, "phase", None), "value", None)
        return bool(
            getattr(telemetry, "zero_latched", False)
            and getattr(telemetry, "body_hold_included", False)
            and getattr(telemetry, "mobility_included", False)
            and mobility is not None
            and bool(getattr(mobility, "is_zero", False))
            and phase not in {"DISCONNECTED", "CLOSED"}
        )

    def confirm_persistent_support(self) -> bool:
        """Confirm either the untouched source or destination zero/body hold."""

        if not self.robot_touched:
            self.persistent_support_confirmed = True
            self.support_owner = "not_touched"
            return True

        if self.destination_commanded or self.destination_steady:
            try:
                self.controller.ensure_persistent_zero_body_hold()
                telemetry = self.controller.telemetry()
                if self._telemetry_confirms_zero_body_hold(telemetry):
                    self.persistent_support_confirmed = True
                    self.support_owner = "destination_zero_body_hold"
                    self.last_hold_error = None
                    self.hold_unrecoverable = False
                    self._unrecoverable_since_s = None
                    return True
                self._note_hold_failure(
                    "destination telemetry did not confirm zero mobility plus body hold"
                )
            except BaseException as exc:  # containment must include interrupts here
                self._note_hold_failure(f"destination hold confirmation failed: {exc}")

        if not self.destination_commanded:
            self.persistent_support_confirmed = True
            self.support_owner = "standalone_ready_session_not_commanded"
            self.last_hold_error = None
            return True
        return False

    def request_shutdown_hold(self, *, next_owner: str) -> bool:
        """First request holds; the second request is an explicit forced cancel."""

        self.interrupt_count += 1
        if self.interrupt_count == 1:
            if not self.confirm_persistent_support():
                self.block_until_escape_is_safe()
                return True
            offer = self.controller.transfer_owner(next_owner)
            self.handoff_offer_created = True
            self.successor_acknowledged = bool(getattr(offer, "acknowledged", False))
            return False
        self.controller.close(force=True)
        self.forced_cancel = True
        return True

    def block_until_escape_is_safe(self) -> None:
        """Never unwind uncertain carried-load ownership without explicit override.

        The wait is unbounded while the zero/body hold is still recoverable.  An
        expired or cancelled command stream can never be re-held, so that case
        is bounded by ``unrecoverable_timeout_s`` and then reported instead of
        spinning forever.
        """

        while not (self.successor_acknowledged or self.forced_cancel):
            try:
                if not self.persistent_support_confirmed:
                    self.confirm_persistent_support()
                if self.persistent_support_confirmed and not self.handoff_offer_created:
                    offer = self.controller.transfer_owner(
                        "runtime-fault-successor-required"
                    )
                    self.handoff_offer_created = True
                    self.successor_acknowledged = bool(
                        getattr(offer, "acknowledged", False)
                    )
                telemetry = self.controller.telemetry()
                phase = getattr(getattr(telemetry, "phase", None), "value", None)
                if phase == "HANDOFF_ACKNOWLEDGED":
                    self.successor_acknowledged = True
                    return
                if self.escape_deadline_reached():
                    print(
                        "DANGER: the RB-Y1 command stream is unrecoverable after "
                        f"{self.unrecoverable_elapsed_s():.1f}s, so no zero/body "
                        "hold can be re-established from this process. Carried-load "
                        "support is NOT guaranteed; secure the load manually. "
                        f"detail={self.last_hold_error}",
                        file=sys.stderr,
                        flush=True,
                    )
                    return
                state = (
                    "successor acknowledgement pending"
                    if self.persistent_support_confirmed
                    else "carried-load support unconfirmed"
                )
                progress = (
                    ""
                    if not self.hold_unrecoverable
                    else (
                        " unrecoverable for "
                        f"{self.unrecoverable_elapsed_s():.1f}s of "
                        f"{self.unrecoverable_timeout_s:.0f}s"
                    )
                )
                print(
                    f"DANGER: {state}; runtime remains alive.{progress} A second "
                    f"Ctrl-C forces cancellation. detail={self.last_hold_error}",
                    file=sys.stderr,
                )
                time.sleep(0.20)
            except KeyboardInterrupt:
                try:
                    self.controller.close(force=True)
                except BaseException as exc:
                    self.last_hold_error = (
                        "forced cancellation failed; runtime remains alive: " f"{exc}"
                    )
                    continue
                self.forced_cancel = True
                return
            except BaseException as exc:
                # This is already the last containment boundary.  Telemetry,
                # transfer, logging, and SDK cleanup failures must not unwind it.
                self._note_hold_failure(f"containment operation failed: {exc}")
                if self.escape_deadline_reached():
                    print(
                        "DANGER: containment cannot recover the RB-Y1 command "
                        f"stream after {self.unrecoverable_elapsed_s():.1f}s. "
                        "Carried-load support is NOT guaranteed; secure the load "
                        f"manually. detail={self.last_hold_error}",
                        file=sys.stderr,
                        flush=True,
                    )
                    return
                try:
                    time.sleep(0.20)
                except KeyboardInterrupt:
                    try:
                        self.controller.close(force=True)
                    except BaseException as close_exc:
                        self.last_hold_error = (
                            "forced cancellation failed; runtime remains alive: "
                            f"{close_exc}"
                        )
                        continue
                    self.forced_cancel = True
                    return
                except BaseException:
                    # Preserve the non-unwinding contract even if the host sleep
                    # primitive itself is interrupted by an unusual BaseException.
                    continue


@dataclass(slots=True)
class LiveFrameGate:
    """Reject stale, buffered, cross-clock, duplicate, or reversing RGB-D frames."""

    maximum_capture_age_s: float
    maximum_rgb_depth_timestamp_skew_s: float
    _last_depth_frame_number: int | None = None
    _last_color_frame_number: int | None = None
    _last_depth_timestamp_ms: float | None = None
    _last_color_timestamp_ms: float | None = None

    @staticmethod
    def _comparable_host_clock_domain(value: Any) -> str | None:
        text = str(value).strip().lower()
        if "global_time" in text:
            return "global_time"
        if "system_time" in text:
            return "system_time"
        return None

    def __post_init__(self) -> None:
        for name in (
            "maximum_capture_age_s",
            "maximum_rgb_depth_timestamp_skew_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            setattr(self, name, value)
        if self.maximum_capture_age_s > 0.20 + 1e-12:
            raise ValueError("maximum_capture_age_s cannot exceed 0.20 seconds")
        if self.maximum_rgb_depth_timestamp_skew_s > 0.05 + 1e-12:
            raise ValueError(
                "maximum_rgb_depth_timestamp_skew_s cannot exceed 0.05 seconds"
            )

    def validate(self, frame: Any, *, wall_time_ns: int | None = None) -> float:
        depth_number = int(frame.depth_frame_number)
        color_number = int(frame.color_frame_number)
        depth_timestamp_ms = float(frame.depth_timestamp_ms)
        color_timestamp_ms = float(frame.color_timestamp_ms)
        system_timestamp_ns = getattr(frame, "system_timestamp_ns", None)
        metadata = getattr(frame, "frame_metadata", {})
        if not isinstance(metadata, Mapping):
            raise RuntimeError("D435 frame metadata is unavailable")
        depth_domain = self._comparable_host_clock_domain(
            metadata.get("depth_timestamp_domain")
        )
        color_domain = self._comparable_host_clock_domain(
            metadata.get("color_timestamp_domain")
        )
        if depth_domain is None or color_domain is None:
            raise RuntimeError(
                "D435 live timestamps must use GLOBAL_TIME or SYSTEM_TIME; "
                "hardware/unknown clock frames cannot prove sensor-to-host age"
            )
        if depth_domain != color_domain:
            raise RuntimeError(
                "D435 RGB/Depth timestamps use different clock domains and are "
                "not comparable"
            )
        if depth_number < 0 or color_number < 0:
            raise RuntimeError("D435 frame numbers must be nonnegative")
        if not all(
            math.isfinite(value) for value in (depth_timestamp_ms, color_timestamp_ms)
        ):
            raise RuntimeError("D435 RGB-D timestamps must be finite")
        if system_timestamp_ns is None:
            raise RuntimeError("D435 capture receipt timestamp is unavailable")
        capture_ns = int(system_timestamp_ns)
        current_ns = time.time_ns() if wall_time_ns is None else int(wall_time_ns)
        capture_age_s = (current_ns - capture_ns) / 1e9
        if (
            not math.isfinite(capture_age_s)
            or capture_age_s < 0.0
            or capture_age_s > self.maximum_capture_age_s
        ):
            raise RuntimeError(
                f"D435 frame capture age {capture_age_s:.3f}s exceeds "
                f"[0, {self.maximum_capture_age_s:.3f}]s"
            )

        # GLOBAL_TIME and SYSTEM_TIME are expressed against the OS system clock.
        # Comparing them with the current wall clock detects a progressing but
        # buffered sequence that a post-wait receipt timestamp alone cannot see.
        current_wall_ms = current_ns / 1e6
        future_tolerance_s = self.maximum_rgb_depth_timestamp_skew_s
        sensor_to_host_ages_s: list[float] = []
        for name, timestamp_ms in (
            ("depth", depth_timestamp_ms),
            ("color", color_timestamp_ms),
        ):
            sensor_to_host_age_s = (current_wall_ms - timestamp_ms) / 1000.0
            if (
                not math.isfinite(sensor_to_host_age_s)
                or sensor_to_host_age_s < -future_tolerance_s
                or sensor_to_host_age_s > self.maximum_capture_age_s
            ):
                raise RuntimeError(
                    f"D435 {name} sensor-to-host age {sensor_to_host_age_s:.3f}s "
                    f"exceeds [-{future_tolerance_s:.3f}, "
                    f"{self.maximum_capture_age_s:.3f}]s"
                )
            sensor_to_host_ages_s.append(max(0.0, sensor_to_host_age_s))

        timestamp_skew_s = abs(depth_timestamp_ms - color_timestamp_ms) / 1000.0
        if timestamp_skew_s > self.maximum_rgb_depth_timestamp_skew_s:
            raise RuntimeError(
                f"RGB/Depth timestamp skew {timestamp_skew_s:.3f}s exceeds "
                f"{self.maximum_rgb_depth_timestamp_skew_s:.3f}s"
            )
        for name, current, previous in (
            ("depth frame number", depth_number, self._last_depth_frame_number),
            ("color frame number", color_number, self._last_color_frame_number),
        ):
            if previous is not None and current <= previous:
                raise RuntimeError(
                    f"D435 {name} is duplicate or non-monotonic: "
                    f"previous={previous} current={current}"
                )
        for name, current, previous in (
            ("depth timestamp", depth_timestamp_ms, self._last_depth_timestamp_ms),
            ("color timestamp", color_timestamp_ms, self._last_color_timestamp_ms),
        ):
            if previous is not None and current <= previous:
                raise RuntimeError(
                    f"D435 {name} is duplicate or non-monotonic: "
                    f"previous={previous:.3f} current={current:.3f}"
                )

        self._last_depth_frame_number = depth_number
        self._last_color_frame_number = color_number
        self._last_depth_timestamp_ms = depth_timestamp_ms
        self._last_color_timestamp_ms = color_timestamp_ms
        return max(capture_age_s, *sensor_to_host_ages_s)


def _section(root: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = root.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"pallet config section {name!r} must be an object")
    return value


def _matrix(value: Any, name: str, shape: tuple[int, int]) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite {shape[0]}x{shape[1]} matrix")
    return result


def configured_T_base_from_depth(root: Mapping[str, Any]) -> np.ndarray:
    """Compose the replay/dry-run transform and apply the correction once."""

    calibration = _section(root, "calibration")
    transform = compose_base_from_depth(
        calibration.get("T_base_from_head_ready_audit"),
        calibration.get("T_head_from_color"),
        calibration.get("E_color_from_depth"),
    )
    if transform is None:
        raise ValueError("configured nominal base-from-depth chain is incomplete")
    result = np.array(transform, dtype=np.float64, copy=True)
    correction = np.asarray(
        calibration.get("base_translation_correction_m", (0.0, 0.0, 0.0)),
        dtype=np.float64,
    )
    if correction.shape != (3,) or not np.all(np.isfinite(correction)):
        raise ValueError(
            "base_translation_correction_m must be a finite length-3 vector"
        )
    result[:3, 3] += correction
    return result


def measured_T_base_from_depth(
    root: Mapping[str, Any],
    T_base_from_head: Any,
) -> np.ndarray:
    """Compose a live transform from fresh ready-posture FK."""

    calibration = _section(root, "calibration")
    transform = compose_base_from_depth(
        _matrix(T_base_from_head, "fresh T_base_from_head", (4, 4)),
        calibration.get("T_head_from_color"),
        calibration.get("E_color_from_depth"),
    )
    if transform is None:
        raise ValueError("live base-from-depth chain is incomplete")
    result = np.array(transform, dtype=np.float64, copy=True)
    correction = np.asarray(
        calibration.get("base_translation_correction_m", (0.0, 0.0, 0.0)),
        dtype=np.float64,
    )
    if correction.shape != (3,) or not np.all(np.isfinite(correction)):
        raise ValueError(
            "base_translation_correction_m must be a finite length-3 vector"
        )
    result[:3, 3] += correction
    return result


def _extrinsic_rotation(extrinsic: Any) -> np.ndarray:
    values = np.asarray(getattr(extrinsic, "rotation", ()), dtype=np.float64)
    if values.shape != (9,) or not np.all(np.isfinite(values)):
        raise ValueError("D435 factory extrinsic rotation is missing or non-finite")
    order = (
        "F"
        if getattr(extrinsic, "rotation_storage", "column_major") == "column_major"
        else "C"
    )
    rotation = values.reshape(3, 3, order=order)
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-4):
        raise ValueError("D435 factory extrinsic rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-4):
        raise ValueError("D435 factory extrinsic rotation determinant is not +1")
    translation = np.asarray(getattr(extrinsic, "translation_m", ()), dtype=np.float64)
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError("D435 factory extrinsic translation is missing or non-finite")
    return rotation


def validate_live_camera_profile(
    metadata: Mapping[str, Any],
    root: Mapping[str, Any],
) -> LiveCameraContract:
    """Fail closed unless the negotiated D435 contract is exact."""

    camera = _section(root, "camera")
    camera_name = str(metadata.get("camera_name", "")).strip()
    required_name = str(camera.get("required_product_name_contains", "D435")).strip()
    if not camera_name or required_name.lower() not in camera_name.lower():
        raise ValueError(
            f"live pallet requires a D435; negotiated product is {camera_name!r}"
        )
    camera_serial = str(metadata.get("camera_serial", "")).strip()
    expected_serial = str(camera.get("serial", "")).strip()
    if not expected_serial or camera_serial != expected_serial:
        raise ValueError(
            "live D435 serial does not match the nominal registration: "
            f"expected={expected_serial!r} actual={camera_serial!r}"
        )

    depth_profile = metadata.get("depth_profile")
    color_profile = metadata.get("color_profile")
    if depth_profile is None or color_profile is None:
        raise ValueError("negotiated D435 depth/color profiles are unavailable")
    expected_depth = _section(camera, "depth")
    expected_color = _section(camera, "color")
    for name, profile, expected in (
        ("depth", depth_profile, expected_depth),
        ("color", color_profile, expected_color),
    ):
        intrinsics = getattr(profile, "intrinsics", None)
        if not isinstance(intrinsics, CameraIntrinsics):
            raise ValueError(f"negotiated {name} intrinsics are unavailable")
        actual = (
            intrinsics.width,
            intrinsics.height,
            intrinsics.fps,
            str(getattr(profile, "format", "")).lower(),
        )
        required = (
            int(expected.get("width", -1)),
            int(expected.get("height", -1)),
            int(expected.get("fps", -1)),
            str(expected.get("format", "")).lower(),
        )
        if actual != required:
            raise ValueError(
                f"negotiated {name} profile mismatch: required={required} actual={actual}"
            )
        if not (0.0 <= intrinsics.cx < intrinsics.width):
            raise ValueError(f"{name} principal point cx is outside the image")
        if not (0.0 <= intrinsics.cy < intrinsics.height):
            raise ValueError(f"{name} principal point cy is outside the image")

    _extrinsic_rotation(metadata.get("depth_to_color"))
    _extrinsic_rotation(metadata.get("color_to_depth"))
    depth_scale = float(metadata.get("depth_scale_m", math.nan))
    if not math.isfinite(depth_scale) or depth_scale <= 0.0:
        raise ValueError("negotiated D435 depth scale is invalid")
    return LiveCameraContract(
        camera_name=camera_name,
        camera_serial=camera_serial,
        depth_scale_m=depth_scale,
        depth_intrinsics=depth_profile.intrinsics,
        color_intrinsics=color_profile.intrinsics,
    )


def _nominal_held_pose(root: Mapping[str, Any]) -> HeldPoseProxy:
    held = _section(root, "held_box")
    right = np.asarray(held["nominal_ready_right_eef_base_xyz_m"], dtype=np.float64)
    left = np.asarray(held["nominal_ready_left_eef_base_xyz_m"], dtype=np.float64)
    return _held_pose_from_eefs(
        right,
        left,
        source="nominal_unverified_ready_eef_box_offset_dry_run",
        center_offset_base=held.get(
            "center_offset_from_eef_midpoint_m", (0.0, 0.0, 0.0)
        ),
        yaw_offset_rad=math.radians(
            float(held.get("long_axis_from_eef_line_offset_deg", 0.0))
        ),
    )


def _fixed_ready_held_pose(
    root: Mapping[str, Any],
    right_eef: Any,
    left_eef: Any,
) -> HeldPoseProxy:
    """Recover the held center/yaw from fresh EEFs at the fixed ready pose."""

    held = _section(root, "held_box")
    return _held_pose_from_eefs(
        right_eef,
        left_eef,
        source="fresh_dual_eef_fixed_ready_nominal_box_offset",
        center_offset_base=held.get(
            "center_offset_from_eef_midpoint_m", (0.0, 0.0, 0.0)
        ),
        yaw_offset_rad=math.radians(
            float(held.get("long_axis_from_eef_line_offset_deg", 0.0))
        ),
    )


def _held_pose_from_eefs(
    right_eef: Any,
    left_eef: Any,
    *,
    source: str,
    center_offset_base: Any = (0.0, 0.0, 0.0),
    yaw_offset_rad: float = 0.0,
) -> HeldPoseProxy:
    right = np.asarray(right_eef, dtype=np.float64)
    left = np.asarray(left_eef, dtype=np.float64)
    if right.shape == (4, 4):
        right = right[:3, 3]
    if left.shape == (4, 4):
        left = left[:3, 3]
    if (
        right.shape != (3,)
        or left.shape != (3,)
        or not np.all(np.isfinite([right, left]))
    ):
        raise ValueError("fresh right/left EEF positions must be finite XYZ vectors")
    separation = left - right
    if np.linalg.norm(separation[:2]) <= 1e-6:
        raise ValueError("right/left EEF XY separation is degenerate")
    offset = np.asarray(center_offset_base, dtype=np.float64)
    if offset.shape != (3,) or not np.all(np.isfinite(offset)):
        raise ValueError(
            "held-box center offset must be a finite base-frame XYZ vector"
        )
    if not math.isfinite(float(yaw_offset_rad)):
        raise ValueError("held-box yaw offset must be finite")
    center = 0.5 * (right + left) + offset
    yaw = math.atan2(float(separation[1]), float(separation[0])) + float(yaw_offset_rad)
    return HeldPoseProxy(tuple(float(value) for value in center), yaw, source)


def _held_hint(root: Mapping[str, Any], proxy: HeldPoseProxy) -> HeldBoxHint:
    held = _section(root, "held_box")
    size = tuple(
        float(value) for value in held.get("nominal_size_m", (0.4, 0.253, 0.16))
    )
    return HeldBoxHint(
        center_base=np.asarray(proxy.center_base_xyz_m, dtype=np.float64),
        yaw_base_rad=proxy.yaw_base_rad,
        eef_proxy_z_base_m=proxy.center_base_xyz_m[2],
        footprint_size_m=(size[0], size[1]),
    )


def _servo_measurement(
    scene: PalletSceneObservation,
    reference: Slot1HoleReference,
    timestamp_s: float,
    geometry: PalletGeometry | None = None,
) -> PalletServoObservation:
    stack = scene.stack
    if stack.valid:
        if (
            stack.center_base is None
            or stack.yaw_base_rad is None
            or stack.axis_branch is None
        ):
            return PalletServoObservation.invalid(
                timestamp_s,
                "complete_hole_geometry_incomplete",
                reference_source=reference.reference_source,
            )
        if stack.axis_branch != reference.axis_branch:
            return PalletServoObservation.invalid(
                timestamp_s,
                "slot1_hole_reference_axis_branch_mismatch",
                reference_source=reference.reference_source,
            )
        return PalletServoObservation(
            timestamp_s=timestamp_s,
            current_observed_feature_center_base=tuple(
                float(value) for value in stack.center_base[:2]
            ),
            current_observed_feature_yaw_base_rad=float(stack.yaw_base_rad),
            demonstrated_body_reference_center_base=reference.center_base_xy_m,
            demonstrated_body_reference_yaw_base_rad=reference.yaw_base_rad,
            axis_branch=stack.axis_branch,
            reference_source=reference.reference_source,
        )

    coarse = scene.coarse
    proxy_center = (
        None
        if geometry is None or coarse is None
        else coarse.fixed_outer_center_base(geometry.outer_size_m)
    )
    if (
        coarse is not None
        and coarse.valid
        and proxy_center is not None
        and coarse.yaw_base_rad is not None
    ):
        return PalletServoObservation(
            timestamp_s=timestamp_s,
            current_observed_feature_center_base=(
                float(proxy_center[0]),
                float(proxy_center[1]),
            ),
            current_observed_feature_yaw_base_rad=float(coarse.yaw_base_rad),
            demonstrated_body_reference_center_base=reference.center_base_xy_m,
            demonstrated_body_reference_yaw_base_rad=reference.yaw_base_rad,
            axis_branch=reference.axis_branch,
            reference_source=reference.reference_source,
        )

    rejection_reasons = tuple(stack.rejection_reasons)
    if coarse is not None:
        rejection_reasons += tuple(coarse.rejection_reasons)
    return PalletServoObservation.invalid(
        timestamp_s,
        *(rejection_reasons or ("fine_geometry_unavailable",)),
        reference_source=reference.reference_source,
    )




def _scene_has_metric_alignment_feature(
    scene: PalletSceneObservation,
    geometry: PalletGeometry,
) -> bool:
    return _servo_measurement_source(scene, geometry) != "unavailable"


def _controller_scene_sample(
    scene: PalletSceneObservation,
    held_proxy: HeldPoseProxy,
    *,
    frame_id: int,
    accepted_observation_sequence: int,
    capture_timestamp_s: float,
    accepted_monotonic_s: float,
    maximum_box_height_m: float,
    box_bottom_uncertainty_m: float,
) -> dict[str, Any]:
    held = scene.held_top
    stack = scene.stack
    coarse = scene.coarse
    stack_top_z = stack.plane_height_base_m
    stack_top_uncertainty = stack.quality.get(
        "stack_plane_p95_residual_m", math.nan
    )
    stack_top_source = {
        "fixed_outer_l_corner": "metric_fixed_outer_l_corner_plane",
        "inner_opening": "metric_inner_opening_plane",
    }.get(stack.stack_se2_source, "metric_stack_plane_candidate")
    if (
        stack_top_z is None
        and coarse is not None
        and (coarse.valid or coarse.forward_acquisition_valid)
        and coarse.plane_height_base_m is not None
    ):
        stack_top_z = coarse.plane_height_base_m
        stack_top_uncertainty = coarse.plane_p95_residual_m
        stack_top_source = (
            "metric_coarse_l_corner_plane"
            if coarse.valid
            else "metric_forward_edge_pair_plane"
        )
    box_bottom_z = (
        float(held_proxy.center_base_xyz_m[2])
        - 0.5 * float(maximum_box_height_m)
    )
    capture_age_at_acceptance_s = (
        float(accepted_monotonic_s) - float(capture_timestamp_s)
    )
    return {
        "frame_id": int(frame_id),
        "accepted_observation_sequence": int(accepted_observation_sequence),
        "capture_timestamp_s": float(capture_timestamp_s),
        "accepted_monotonic_s": float(accepted_monotonic_s),
        "capture_age_at_acceptance_s": float(capture_age_at_acceptance_s),
        "fresh_at_acceptance": bool(
            0.0 <= capture_age_at_acceptance_s
            < _MAX_LIVE_CONTROL_RESULT_AGE_S
        ),
        "held_top_distinct_from_stack": bool(
            held is not None and held.valid and held.distinct_from_stack
        ),
        "held_top_z_base_m": None if held is None else held.top_plane_z_base_m,
        "held_top_uncertainty_m": (
            None if held is None else held.top_plane_z_uncertainty_m
        ),
        "stack_top_z_base_m": stack_top_z,
        "stack_top_uncertainty_m": stack_top_uncertainty,
        "stack_top_source": stack_top_source,
        "held_box_bottom_z_base_m": box_bottom_z,
        "held_box_bottom_uncertainty_m": float(box_bottom_uncertainty_m),
        "held_box_pose_source": held_proxy.source,
    }


def _update_scene_window(
    scene_window: deque[Any],
    *,
    frame_result_fresh: bool,
    fresh_sample: Any | None,
) -> bool:
    """Keep accepted evidence across an isolated late processing result.

    A late result still forces exact-zero mobility for that cycle.  It must not,
    however, erase earlier evidence that was fresh when accepted: raw D435 frame
    drops are allowed, and the clearance evaluator independently rejects old or
    over-long evidence windows.  The next fresh observation can therefore
    resume from the retained accepted sequence instead of paying another full
    five-frame acquisition delay.
    """

    if not frame_result_fresh or fresh_sample is None:
        return False
    scene_window.append(fresh_sample)
    return True


def _held_box_height_bounds(root: Mapping[str, Any]) -> tuple[float, float, float]:
    held = _section(root, "held_box")
    nominal_size = tuple(
        float(value) for value in held.get("nominal_size_m", (0.400, 0.253, 0.160))
    )
    if len(nominal_size) != 3:
        raise ValueError("held_box.nominal_size_m must contain three values")
    nominal_height = float(nominal_size[2])
    minimum_height = float(held.get("minimum_height_m", 0.156))
    maximum_height = float(held.get("maximum_height_m", 0.164))
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (minimum_height, nominal_height, maximum_height)
    ):
        raise ValueError("held-box height bounds must be finite and positive")
    if minimum_height > nominal_height or nominal_height > maximum_height:
        raise ValueError(
            "held-box height bounds must satisfy minimum <= nominal <= maximum"
        )
    return minimum_height, nominal_height, maximum_height


def _predict_box_bottom_gap(
    root: Mapping[str, Any],
    scene: PalletSceneObservation,
    held_proxy: HeldPoseProxy | None = None,
) -> tuple[float | None, float | None, dict[str, Any]]:
    """Compute metric box-bottom clearance from bilateral EEF FK and stack plane."""

    minimum_height, nominal_height, maximum_height = _held_box_height_bounds(root)
    height_half_range_m = 0.5 * (maximum_height - minimum_height)
    stack = scene.stack
    diagnostics: dict[str, Any] = {
        "source": "bilateral_eef_fk_box_bottom_minus_stack_plane",
        "minimum_box_height_m": minimum_height,
        "nominal_box_height_m": nominal_height,
        "maximum_box_height_m": maximum_height,
        "height_half_range_m": height_half_range_m,
        "valid": False,
    }
    if held_proxy is None:
        diagnostics["reason"] = "bilateral_eef_fk_box_pose_unavailable"
        return None, None, diagnostics
    if not stack.valid or stack.plane_height_base_m is None:
        diagnostics["reason"] = (
            "stack_plane_invalid"
            if stack.valid
            else ";".join(stack.rejection_reasons)
        )
        return None, None, diagnostics
    stack_residual = stack.quality.get("stack_plane_p95_residual_m")
    if stack_residual is None or not math.isfinite(float(stack_residual)):
        diagnostics["reason"] = "stack_plane_residual_unavailable"
        return None, None, diagnostics

    center_z = float(held_proxy.center_base_xyz_m[2])
    fixed_ready_uncertainty_m = float(
        _section(root, "held_box").get(
            "fixed_ready_box_bottom_uncertainty_m",
            0.015,
        )
    )
    if (
        not math.isfinite(center_z)
        or not math.isfinite(fixed_ready_uncertainty_m)
        or fixed_ready_uncertainty_m < 0.0
    ):
        diagnostics["reason"] = "bilateral_eef_fk_box_bottom_invalid"
        return None, None, diagnostics
    stack_top_z = float(stack.plane_height_base_m)
    stack_uncertainty = max(0.0, float(stack_residual))
    nominal_box_bottom_z = center_z - 0.5 * nominal_height
    box_bottom_lower_bound_z = (
        center_z - 0.5 * maximum_height - fixed_ready_uncertainty_m
    )
    stack_top_upper_bound_z = stack_top_z + stack_uncertainty
    gap_m = nominal_box_bottom_z - stack_top_z
    uncertainty_m = (
        nominal_box_bottom_z
        - box_bottom_lower_bound_z
        + stack_uncertainty
    )
    clearance_lower_bound_m = box_bottom_lower_bound_z - stack_top_upper_bound_z
    diagnostics.update(
        {
            "valid": True,
            "held_box_pose_source": held_proxy.source,
            "held_box_center_z_base_m": center_z,
            "box_bottom_z_nominal_m": nominal_box_bottom_z,
            "box_bottom_z_lower_bound_m": box_bottom_lower_bound_z,
            "stack_top_z_base_m": stack_top_z,
            "stack_top_z_upper_bound_m": stack_top_upper_bound_z,
            "fixed_ready_box_bottom_uncertainty_m": fixed_ready_uncertainty_m,
            "stack_plane_p95_residual_m": stack_uncertainty,
            "clearance_lower_bound_m": clearance_lower_bound_m,
            "predicted_box_bottom_gap_m": gap_m,
            "predicted_box_bottom_gap_uncertainty_m": uncertainty_m,
        }
    )
    return gap_m, uncertainty_m, diagnostics


def _controller_arm_mode_string(placement_telemetry: Any | None) -> str:
    if placement_telemetry is None:
        return ""
    mode = getattr(placement_telemetry, "arm_mode", "")
    return str(getattr(mode, "value", mode))




def _dispatch_placement_fault_hold_if_needed(
    controller: _ControllerLike,
    placement: PlacementOutput,
    *,
    lowering_started: bool,
    release_started: bool,
) -> str | None:
    if not placement.faulted or not (lowering_started or release_started):
        return None
    controller.fail_closed_cartesian_placement_hold(reason=placement.reason)
    return "fail_closed_cartesian_placement_hold"


def _placement_input(
    root: Mapping[str, Any],
    scene: PalletSceneObservation,
    controller: _ControllerLike,
    *,
    now_s: float,
    gap_observation_timestamp_s: float,
    gap_observation_sequence: int,
    decision: PalletServoOutput,
    zero_acknowledged: bool,
    stationary: bool,
    allow_vision_geometry_release: bool,
) -> tuple[PlacementInput, dict[str, Any]]:
    measured_state = controller.get_measured_state()
    right_eef, left_eef = controller.get_measured_eef_transforms()
    telemetry = controller.placement_telemetry()
    held_proxy = _fixed_ready_held_pose(root, right_eef, left_eef)
    gap_m, gap_uncertainty_m, gap_diagnostics = _predict_box_bottom_gap(
        root,
        scene,
        held_proxy,
    )
    feedback_timestamp_s = float(
        getattr(measured_state, "received_monotonic_s", now_s)
    )
    feedback_sequence = int(getattr(measured_state, "sequence", 0))
    feedback_stale_s = float(
        _section(root, "placement").get("feedback_stale_s", 0.25)
    )
    feedback_age_s = now_s - feedback_timestamp_s
    measured_state_fresh = bool(
        feedback_sequence > 0
        and math.isfinite(feedback_age_s)
        and -1e-12 <= feedback_age_s <= feedback_stale_s
    )
    box_bottom_z = gap_diagnostics.get("box_bottom_z_nominal_m")
    box_bottom_lower = gap_diagnostics.get("box_bottom_z_lower_bound_m")
    stack_top_z = gap_diagnostics.get("stack_top_z_base_m")
    stack_top_upper = gap_diagnostics.get("stack_top_z_upper_bound_m")
    box_bottom_uncertainty = (
        None
        if box_bottom_z is None or box_bottom_lower is None
        else max(0.0, float(box_bottom_z) - float(box_bottom_lower))
    )
    stack_top_uncertainty = (
        None
        if stack_top_z is None or stack_top_upper is None
        else max(0.0, float(stack_top_upper) - float(stack_top_z))
    )
    input_sample = PlacementInput(
        now_s=now_s,
        feedback_timestamp_s=feedback_timestamp_s,
        right_eef_base=right_eef,
        left_eef_base=left_eef,
        arrived_hold=decision.state is PalletServoState.ARRIVED_HOLD,
        post_zero_wheel_stop=bool(stationary and zero_acknowledged),
        zero_command_ack=bool(zero_acknowledged),
        measured_state_fresh=measured_state_fresh,
        controller_stream_healthy=bool(getattr(telemetry, "stream_running", False)),
        controller_arm_mode=_controller_arm_mode_string(telemetry),
        controller_target_ack=bool(getattr(telemetry, "target_acknowledged", False)),
        ready_posture_verified=bool(
            getattr(telemetry, "ready_posture_verified", False)
        ),
        right_target_base=getattr(telemetry, "right_T_base_eef_target", None),
        left_target_base=getattr(telemetry, "left_T_base_eef_target", None),
        allow_vision_geometry_release=bool(allow_vision_geometry_release),
        predicted_box_bottom_gap_m=gap_m,
        predicted_box_bottom_gap_uncertainty_m=gap_uncertainty_m,
        gap_observation_timestamp_s=gap_observation_timestamp_s,
        gap_observation_sequence=gap_observation_sequence,
        box_bottom_z_base_m=box_bottom_z,
        box_bottom_z_uncertainty_m=box_bottom_uncertainty,
        stack_top_z_base_m=stack_top_z,
        stack_top_uncertainty_m=stack_top_uncertainty,
        stack_plane_z_base_m=stack_top_z,
        stack_plane_uncertainty_m=stack_top_uncertainty,
        stack_plane_timestamp_s=gap_observation_timestamp_s,
        stack_plane_sequence=gap_observation_sequence,
        bilateral_eef_timestamp_s=feedback_timestamp_s,
        bilateral_eef_state_sequence=(
            feedback_sequence if feedback_sequence > 0 else None
        ),
        descent_plan_source=(
            "bilateral_eef_fk_box_bottom_minus_metric_stack_plane"
        ),
        demonstrated_place_pose=bool(
            getattr(getattr(controller, "config", None), "place_pose", None) is not None
        ),
    )
    diagnostics = {
        "gap_prediction": gap_diagnostics,
        "placement_telemetry": _placement_telemetry_payload(telemetry),
    }
    return input_sample, diagnostics


def _annotate_placement_output(
    output: PalletServoOutput,
    placement: PlacementOutput | None,
    diagnostics: Mapping[str, Any] | None = None,
) -> PalletServoOutput:
    if placement is None:
        return output
    merged = dict(output.diagnostics)
    merged["placement"] = {
        "state": placement.state.value,
        "request": placement.request.value,
        "reason": placement.reason,
        "done": placement.done,
        "faulted": placement.faulted,
        "release_authorized": placement.release_authorized,
        "diagnostics": dict(placement.diagnostics),
        "runtime": dict(diagnostics or {}),
    }
    reason = f"{output.reason}; placement:{placement.reason}"
    return PalletServoOutput(
        command=output.command,
        state=output.state,
        arrived=output.arrived,
        hold_body=output.hold_body,
        measurement_accepted=output.measurement_accepted,
        reason=reason,
        diagnostics=merged,
    )


def _wheel_measurement(
    status: Any,
    now_s: float,
) -> WheelMotionMeasurement | None:
    if (
        not bool(getattr(status, "feedback_fresh", False))
        or getattr(status, "linear_speed_mps", None) is None
        or getattr(status, "angular_speed_radps", None) is None
    ):
        return None
    return WheelMotionMeasurement(
        timestamp_s=now_s,
        linear_speed_mps=float(status.linear_speed_mps),
        angular_speed_radps=float(status.angular_speed_radps),
    )


def _stationary_at_capture(
    wheel_status: Any,
    frame_source_monotonic_s: float,
    decision_now_s: float,
) -> bool:
    """Prove the wheels were already below threshold when this frame was captured."""

    dwell_s = float(getattr(wheel_status, "dwell_s", math.nan))
    frame_s = float(frame_source_monotonic_s)
    now_s = float(decision_now_s)
    return bool(
        getattr(wheel_status, "feedback_fresh", False)
        and getattr(wheel_status, "stopped", False)
        and all(math.isfinite(value) for value in (dwell_s, frame_s, now_s))
        and dwell_s >= 0.0
        and frame_s >= now_s - dwell_s - 1e-12
        and frame_s <= now_s + 1e-12
    )


def _odometry_sample(
    controller: _ControllerLike,
    now_s: float,
) -> tuple[OdometrySample | None, str | None]:
    """Convert fresh measured ``T_odom_base`` into the pure SE(2) contract."""

    try:
        transform, sequence, age_s = controller.get_measured_odometry()
        matrix = np.asarray(transform, dtype=np.float64)
        age = float(age_s)
        if not np.all(np.isfinite(matrix)):
            raise ValueError("T_odom_base must be finite")
        if matrix.shape == (3, 3):
            x_m = float(matrix[0, 2])
            y_m = float(matrix[1, 2])
        elif matrix.shape == (4, 4):
            x_m = float(matrix[0, 3])
            y_m = float(matrix[1, 3])
        else:
            raise ValueError("T_odom_base must be a 3x3 SE(2) or 4x4 transform")
        if not math.isfinite(age) or age < 0.0:
            raise ValueError("odometry age must be finite and non-negative")
        return (
            OdometrySample(
                timestamp_s=now_s - age,
                x_m=x_m,
                y_m=y_m,
                yaw_rad=math.atan2(float(matrix[1, 0]), float(matrix[0, 0])),
                sequence=int(sequence),
            ),
            None,
        )
    except Exception as exc:
        return None, f"odometry_unavailable:{type(exc).__name__}:{exc}"


def _zero_command_acknowledged(controller: _ControllerLike) -> bool:
    """Require the live combined owner to have published an exact-zero packet."""

    telemetry = controller.telemetry()
    mobility = getattr(telemetry, "last_sent_mobility", None)
    return bool(
        getattr(telemetry, "body_hold_included", False)
        and getattr(telemetry, "mobility_included", False)
        and int(getattr(telemetry, "command_sequence", 0)) > 0
        and mobility is not None
        and bool(getattr(mobility, "is_zero", False))
    )


def _acquisition_as_servo_output(
    acquisition: AcquisitionOutput,
    l_gate: LCornerGateStatus,
    hole_gate: HoleGateStatus,
) -> PalletServoOutput:
    """Adapt the coarse owner to the existing display/stream output boundary."""

    if acquisition.state is AcquisitionState.STEP:
        state = PalletServoState.TRACKING
    elif acquisition.state is AcquisitionState.FAULT_HOLD:
        state = PalletServoState.FAULT_HOLD
    elif acquisition.state in {
        AcquisitionState.OBSERVE,
        AcquisitionState.SETTLE,
    }:
        state = PalletServoState.ACQUIRING
    else:
        state = PalletServoState.PERCEPTION_HOLD
    diagnostics: dict[str, object] = {
        "controller_owner": "forward_acquisition",
        "acquisition": acquisition.to_dict(),
        "l_corner_gate": asdict(l_gate),
        "hole_gate": asdict(hole_gate),
        "raw_error_xy_m": None,
        "raw_yaw_error_rad": None,
    }
    return PalletServoOutput(
        command=VelocityCommand(
            acquisition.vx_mps,
            acquisition.vy_mps,
            acquisition.wz_radps,
        ),
        state=state,
        arrived=False,
        hold_body=True,
        measurement_accepted=l_gate.stable or hole_gate.dwell_complete,
        reason=f"acquisition:{acquisition.reason}",
        diagnostics=diagnostics,
    )


def _shutdown_hold_output(reason: str) -> PalletServoOutput:
    return PalletServoOutput(
        command=VelocityCommand(),
        state=PalletServoState.SHUTDOWN_PENDING_HOLD,
        arrived=False,
        hold_body=True,
        measurement_accepted=False,
        reason=reason,
        diagnostics={"controller_owner": "shutdown_hold"},
    )


def _annotate_fine_output(
    output: PalletServoOutput,
    *,
    handoff_started: bool = False,
    measurement_source: str | None = None,
    bridge_diagnostics: Mapping[str, Any] | None = None,
) -> PalletServoOutput:
    diagnostics = dict(output.diagnostics)
    diagnostics["controller_owner"] = "fine_slot1_servo"
    diagnostics["coarse_to_fine_handoff_started"] = bool(handoff_started)
    if measurement_source is not None:
        diagnostics["observed_feature_source"] = str(measurement_source)
    diagnostics["visual_dropout_bridge"] = dict(bridge_diagnostics or {})
    return PalletServoOutput(
        command=output.command,
        state=output.state,
        arrived=output.arrived,
        hold_body=output.hold_body,
        measurement_accepted=output.measurement_accepted,
        reason=("coarse_to_fine_handoff_at_zero" if handoff_started else output.reason),
        diagnostics=diagnostics,
    )


def _placement_zero_hold_output(output: PalletServoOutput) -> PalletServoOutput:
    """Freeze mobile authority after the first placement Cartesian command."""

    diagnostics = dict(output.diagnostics)
    diagnostics["controller_owner"] = "slot1_placement"
    diagnostics["mobile_authority_locked_zero"] = True
    diagnostics["alignment_state_preserved_during_placement"] = output.state.value
    return PalletServoOutput(
        command=VelocityCommand(),
        state=output.state,
        arrived=output.arrived,
        hold_body=True,
        measurement_accepted=output.measurement_accepted,
        reason="placement_active_exact_zero_mobility",
        diagnostics=diagnostics,
    )


def _send_persistent_zero(controller: _ControllerLike) -> None:
    telemetry = controller.telemetry()
    if bool(getattr(telemetry, "zero_latched", False)):
        controller.send_cycle(VelocityCommand(), owner_epoch=controller.owner_epoch)
    else:
        controller.send_zero_mobility_hold(latch=True)


def _live_result_fresh(
    source_timestamp_s: float,
    now_s: float,
    *,
    max_age_s: float = _MAX_LIVE_CONTROL_RESULT_AGE_S,
) -> bool:
    age_s = float(now_s) - float(source_timestamp_s)
    max_age = float(max_age_s)
    if not (
        math.isfinite(age_s)
        and math.isfinite(max_age)
        and max_age > 0.0
        and -1e-9 <= age_s
    ):
        return False
    if math.isclose(max_age, _MAX_LIVE_CONTROL_RESULT_AGE_S, abs_tol=1e-12):
        return bool(age_s + 1e-12 < max_age)
    return bool(
        age_s <= max_age + 1e-12
    )


def _raw_complete_hole_evidence(
    scene: PalletSceneObservation,
    *,
    frame_result_fresh: bool,
) -> tuple[bool, float | None]:
    """Expose raw complete-hole evidence so continuous acquisition can brake."""

    if (
        not frame_result_fresh
        or not scene.stack.valid
        or scene.stack.opening_size_m is None
    ):
        return False, None
    try:
        timestamp_s = float(scene.stack.timestamp_s)
    except (AttributeError, TypeError, ValueError):
        return False, None
    if not math.isfinite(timestamp_s):
        return False, None
    return True, timestamp_s


def _dispatch_live_decision(
    controller: _ControllerLike,
    authority: CoarseFineAuthority,
    owner: PalletControlOwner,
    decision: PalletServoOutput,
    *,
    motion_interlocks_ok: bool,
    source_timestamp_s: float,
    source_max_age_s: float = _MAX_LIVE_CONTROL_RESULT_AGE_S,
    now_s: float | None = None,
) -> str:
    """Submit one sourced proposal, substituting zero for stale live results."""

    sourced = MobilityCommand(
        decision.command.vx_mps,
        decision.command.vy_mps,
        decision.command.wz_radps,
        source_timestamp_s=float(source_timestamp_s),
    )
    authority.assert_publish(owner, sourced)
    if decision.state in {
        PalletServoState.PERCEPTION_HOLD,
        PalletServoState.FAULT_HOLD,
        PalletServoState.ARRIVAL_WHEEL_STOP,
        PalletServoState.ARRIVED_HOLD,
        PalletServoState.SHUTDOWN_PENDING_HOLD,
    }:
        _send_persistent_zero(controller)
        return "state_requires_persistent_zero"
    if sourced.is_zero:
        controller.send_cycle(sourced, owner_epoch=controller.owner_epoch)
        return "exact_zero_decision"

    publish_s = time.monotonic() if now_s is None else float(now_s)
    if not _live_result_fresh(
        source_timestamp_s,
        publish_s,
        max_age_s=source_max_age_s,
    ):
        _send_persistent_zero(controller)
        return "frame_result_stale_selected_zero"
    if decision.diagnostics.get("controller_owner") == "forward_acquisition" and (
        sourced.vx_mps <= 0.0 or sourced.vy_mps != 0.0 or sourced.wz_radps != 0.0
    ):
        raise RuntimeError("forward acquisition emitted a non-forward command")
    if not motion_interlocks_ok:
        _send_persistent_zero(controller)
        return "motion_interlock_selected_zero"
    if bool(getattr(controller.telemetry(), "zero_latched", False)):
        controller.resume_mobility(owner_epoch=controller.owner_epoch)
    controller.send_cycle(sourced, owner_epoch=controller.owner_epoch)
    return "nonzero_proposal_accepted"






def _open_video(path: Path | None, shape: tuple[int, int], fps: int) -> Any | None:
    if path is None:
        return None
    if path.exists():
        raise FileExistsError(f"refusing to overwrite live video: {path}")
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to write the pallet MP4") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = shape
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open pallet MP4 writer: {path}")
    return writer


def _preflight_output_paths(
    log_jsonl: "str | Path | None",
    output_mp4: "str | Path | None",
) -> None:
    """Reject unusable artifact paths before the robot is ever touched.

    ``_open_log`` runs after the combined stream already owns the arms, so a
    refusal there strands a carried load in containment and forces an unsafe
    cancellation; that is how a live run dropped a carton.  The same refusal is
    cheap to make here, before any SDK import or connection.
    """

    for label, value in (("live telemetry", log_jsonl), ("overlay video", output_mp4)):
        if value is None:
            continue
        path = Path(value)
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {label}: {path}")
        parent = path.parent
        if parent.exists() and not parent.is_dir():
            raise NotADirectoryError(f"{label} parent is not a directory: {parent}")
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise OSError(f"cannot create {label} directory {parent}: {exc}") from exc
        probe = parent / f".{path.name}.preflight"
        try:
            with probe.open("x", encoding="utf-8"):
                pass
        except OSError as exc:
            raise OSError(f"cannot write {label} into {parent}: {exc}") from exc
        finally:
            probe.unlink(missing_ok=True)


def _open_log(path: Path | None) -> TextIO | None:
    if path is None:
        return None
    if path.exists():
        raise FileExistsError(f"refusing to overwrite live telemetry: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("x", encoding="utf-8")






def _write_record(stream: TextIO | None, record: Mapping[str, Any]) -> None:
    if stream is None:
        return
    stream.write(
        json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
    )
    stream.flush()


def _gui_available() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _prepare_loaded_ready_actuation(
    controller: _ControllerLike,
    minimum_time_s: float,
    containment: ActuationContainmentState,
) -> None:
    """Start the standalone loaded-ready combined owner at exact zero mobility."""

    controller.connect()
    controller.bootstrap_loaded_slot1_ready(loaded_box_acknowledged=True)

    def mark_ambiguous_transport_boundary() -> None:
        # A transport exception after this point can occur after the robot has
        # accepted the packet, so containment must assume destination ownership.
        containment.mark_robot_touch()
        containment.mark_destination_commanded()

    transition_id = controller.send_ready_transition_once(
        minimum_time_s,
        on_send_attempt=mark_ambiguous_transport_boundary,
    )
    controller.wait_ready_transition_ack(transition_id)
    controller.start_combined_stream()
    containment.mark_destination_steady()
    controller.reverify_wheel_stop_after_stream_start(timeout_s=2.0)


@dataclass(frozen=True)
class _BaseMotionDecision:
    """What one frame decided about the base, and the evidence behind it.

    Every field is produced fresh each frame; nothing here carries over, which is
    why the decision can be a function rather than a stage of a long method.
    """

    acquisition_output: Any
    bridge_diagnostics: Any
    decision: Any
    decision_owner: Any
    decision_source_max_age_s: Any
    decision_source_timestamp_s: Any
    dispatch_result: Any
    grip_result: Any
    hole_status: Any
    l_status: Any
    motion_interlock_reason: Any
    motion_interlocks_ok: Any
    odometry: Any
    odometry_error: Any
    placement_motion_active: Any
    stationary: Any
    stationary_source: Any


def _decide_base_motion(
    *,
    acquisition_config: Any,
    acquisition_servo: Any,
    authority: Any,
    auto_place_slot1: Any,
    controller: Any,
    decision_now_s: Any,
    estimator_config: Any,
    execute: Any,
    frame_result_fresh: Any,
    frame_source_monotonic_s: Any,
    geometry_only_policy_enabled: Any,
    hole_gate: Any,
    l_corner_gate: Any,
    placement_lowering_started: Any,
    placement_release_started: Any,
    placement_sequencer: Any,
    scene: Any,
    scene_window: Any,
    servo: Any,
    servo_bridge: Any,
    shutdown_pending: Any,
    slot1_hole_reference: Any,
) -> _BaseMotionDecision:
    """Decide what the base does this frame: acquire, servo, hold or stop.

    Pure per-frame: the caller owns the loop and the persistent placement flags,
    so a wrong base motion is diagnosed here and nowhere else.
    """

    acquisition_output = None
    bridge_diagnostics = None
    decision = None
    decision_owner = None
    decision_source_max_age_s = None
    decision_source_timestamp_s = None
    dispatch_result = None
    grip_result = None
    hole_status = None
    l_status = None
    motion_interlock_reason = None
    motion_interlocks_ok = None
    odometry = None
    odometry_error = None
    placement_motion_active = None
    stationary = None
    stationary_source = None
    zero_acknowledged = None

    if execute:
        assert controller is not None
        wheel_status = controller.wheel_stop_status()
        wheel = _wheel_measurement(wheel_status, decision_now_s)
        stationary = bool(
            getattr(wheel_status, "feedback_fresh", False)
            and getattr(wheel_status, "stopped", False)
        )
        stationary_at_capture = _stationary_at_capture(
            wheel_status,
            frame_source_monotonic_s,
            decision_now_s,
        )
        stationary_source = "measured_wheel_stop_dwell"
        odometry, odometry_error = _odometry_sample(
            controller,
            decision_now_s,
        )
        zero_acknowledged = _zero_command_acknowledged(controller)
        if scene_window:
            grip_result = controller.evaluate_grip_and_clearance_dwell(
                list(scene_window),
                allow_fixed_ready_geometry_only=(
                    geometry_only_policy_enabled
                ),
            )
            motion_interlocks_ok = bool(
                getattr(grip_result, "passed", False)
            )
            grip_reasons = tuple(
                getattr(grip_result, "reasons", ())
            )
            motion_interlock_reason = (
                "" if motion_interlocks_ok else ";".join(grip_reasons)
            )
        else:
            motion_interlocks_ok = False
            motion_interlock_reason = "clearance_evidence_unavailable"
    else:
        wheel_status = None
        wheel = None
        stationary = True
        stationary_at_capture = True
        stationary_source = "dry_run_assumed_stationary_no_actuation"
        odometry = None
        odometry_error = "dry_run_has_no_robot_odometry"
        zero_acknowledged = True
        motion_interlocks_ok = False
        motion_interlock_reason = "dry_run_no_robot_commands"

    stationary_for_vision = stationary_at_capture and frame_result_fresh
    coarse = scene.coarse
    raw_l_corner_visible = bool(
        frame_result_fresh
        and coarse is not None
        and (
            coarse.valid
            or coarse.forward_acquisition_valid
        )
    )
    raw_metric_proxy_visible = bool(
        frame_result_fresh
        and coarse is not None
        and coarse.fixed_outer_center_base(
            estimator_config.geometry.outer_size_m
        )
        is not None
    )
    raw_l_corner_timestamp_s = (
        float(coarse.timestamp_s)
        if raw_l_corner_visible and coarse is not None
        else None
    )
    raw_hole_visible, raw_hole_timestamp_s = (
        _raw_complete_hole_evidence(
            scene,
            frame_result_fresh=frame_result_fresh,
        )
    )
    l_status = l_corner_gate.update(
        coarse,
        stationary=stationary_for_vision,
    )
    hole_status = hole_gate.update(
        scene,
        stationary=stationary_for_vision,
    )
    acquisition_output: AcquisitionOutput | None = None
    bridge_diagnostics: dict[str, Any] = {
        "odometry_prediction_used": False,
        "dropout_age_s": None,
        "prediction_ttl_s": _MAX_ODOMETRY_PREDICTION_AGE_S,
        "prediction_reason": "fine_servo_not_active",
    }
    decision_source_timestamp_s = frame_source_monotonic_s
    if shutdown_pending:
        decision = _shutdown_hold_output("shutdown_handoff_pending")
        decision_owner = PalletControlOwner.SHUTDOWN_HOLD
    elif authority.owner is PalletControlOwner.FINE_SLOT1_SERVO:
        raw_measurement = (
            _servo_measurement(
                scene,
                slot1_hole_reference,
                frame_source_monotonic_s,
                estimator_config.geometry,
            )
            if frame_result_fresh
            else PalletServoObservation.invalid(
                frame_source_monotonic_s,
                "frame_processing_stale",
                reference_source=(
                    slot1_hole_reference.reference_source
                ),
            )
        )
        measurement, bridge_diagnostics = servo_bridge.resolve(
            raw_measurement,
            odometry,
            now_s=decision_now_s,
        )
        fine_output = servo.update(measurement, decision_now_s, wheel)
        if frame_result_fresh and raw_measurement.valid:
            servo_bridge.commit(
                raw_measurement,
                odometry,
                fine_output,
                now_s=decision_now_s,
            )
        decision_source_timestamp_s = float(measurement.timestamp_s)
        decision = _annotate_fine_output(
            fine_output,
            measurement_source=(
                "odometry_predicted_metric_stack"
                if bridge_diagnostics.get("odometry_prediction_used")
                else _servo_measurement_source(
                    scene,
                    estimator_config.geometry,
                )
            ),
            bridge_diagnostics=bridge_diagnostics,
        )
        decision_owner = PalletControlOwner.FINE_SLOT1_SERVO
    else:
        acquisition_decision = AcquisitionDecision(
            now_s=decision_now_s,
            odometry=odometry,
            # Raw per-frame evidence remains a moving-step safety
            # predicate.  Only the stationary gate may authorize a
            # new step or a coarse-to-fine transition.
            l_corner_visible=raw_l_corner_visible,
            l_corner_stable=l_status.stable,
            l_corner_stationary_frames=l_status.stationary_frames,
            l_corner_window_started_at_s=l_status.window_started_at_s,
            l_corner_timestamp_s=raw_l_corner_timestamp_s,
            l_corner_topology_branch=(
                coarse.topology_branch
                if raw_l_corner_visible and coarse is not None
                else None
            ),
            metric_proxy_visible=raw_metric_proxy_visible,
            metric_proxy_stable=(
                l_status.metric_proxy_stable
                and raw_metric_proxy_visible
            ),
            metric_proxy_timestamp_s=(
                float(coarse.timestamp_s)
                if raw_metric_proxy_visible and coarse is not None
                else None
            ),
            hole_visible=raw_hole_visible,
            hole_visible_timestamp_s=raw_hole_timestamp_s,
            hole_dwell_complete=hole_status.dwell_complete,
            hole_window_started_at_s=(
                hole_status.window_started_at_s
                if hole_status.dwell_complete
                else None
            ),
            hole_timestamp_s=(
                hole_status.window_ended_at_s
                if hole_status.dwell_complete
                else None
            ),
            motion_interlocks_ok=motion_interlocks_ok,
            interlock_reason=motion_interlock_reason,
            wheel_stopped=stationary,
            wheel_timestamp_s=(decision_now_s if stationary else None),
            zero_command_acknowledged=zero_acknowledged,
        )
        acquisition_output = acquisition_servo.update(acquisition_decision)
        if acquisition_output.fine_handoff_requested:
            if acquisition_config.budget_m <= 0.0:
                raise RuntimeError(
                    "zero acquisition budget cannot transfer motion authority"
                )
            if not acquisition_output.is_exact_zero or not stationary:
                raise RuntimeError(
                    "coarse-to-fine handoff requires exact zero and "
                    "measured wheel stop"
                )
            authority.handoff_to_fine(
                acquisition_output,
                zero_command_acknowledged=zero_acknowledged,
                wheel_stopped=stationary,
            )
            decision = _annotate_fine_output(
                servo.start(decision_now_s),
                handoff_started=True,
                measurement_source=_servo_measurement_source(
                    scene,
                    estimator_config.geometry,
                ),
                bridge_diagnostics=bridge_diagnostics,
            )
            decision_owner = PalletControlOwner.FINE_SLOT1_SERVO
        else:
            decision = _acquisition_as_servo_output(
                acquisition_output, l_status, hole_status
            )
            decision_owner = PalletControlOwner.FORWARD_ACQUISITION

    placement_motion_active = bool(
        auto_place_slot1
        and placement_sequencer is not None
        and not shutdown_pending
        and (
            placement_lowering_started
            or placement_release_started
            or placement_sequencer.state
            is not PlacementState.PRE_PLACE_VERIFY
        )
    )
    if placement_motion_active:
        decision = _placement_zero_hold_output(decision)
        decision_owner = PalletControlOwner.FINE_SLOT1_SERVO
        motion_interlocks_ok = True
        motion_interlock_reason = (
            "placement_cartesian_motion_exact_zero_base"
        )

    decision_source_max_age_s = (
        _MAX_ODOMETRY_PREDICTION_AGE_S
        if bridge_diagnostics.get("odometry_prediction_used")
        else _MAX_LIVE_CONTROL_RESULT_AGE_S
    )
    dispatch_result = "dry_run_no_actuation"

    return _BaseMotionDecision(
        acquisition_output=acquisition_output,
        bridge_diagnostics=bridge_diagnostics,
        decision=decision,
        decision_owner=decision_owner,
        decision_source_max_age_s=decision_source_max_age_s,
        decision_source_timestamp_s=decision_source_timestamp_s,
        dispatch_result=dispatch_result,
        grip_result=grip_result,
        hole_status=hole_status,
        l_status=l_status,
        motion_interlock_reason=motion_interlock_reason,
        motion_interlocks_ok=motion_interlocks_ok,
        odometry=odometry,
        odometry_error=odometry_error,
        placement_motion_active=placement_motion_active,
        stationary=stationary,
        stationary_source=stationary_source,
    )

@dataclass(frozen=True)
class _PlacementStep:
    """What the placement sequencer did this frame, and the state it carries on.

    The five carried fields are the whole memory of the placement: whether the
    lowering and the release have been started, when the alignment first became
    ready, and the last output kept for telemetry.
    """

    decision: Any
    dispatch_result: Any
    placement_output: Any
    placement_runtime_diagnostics: Any
    last_placement_output: Any
    last_placement_runtime_diagnostics: Any
    placement_alignment_ready_since_s: Any
    placement_lowering_started: Any
    placement_release_started: Any


def _advance_placement(
    *,
    authority: Any,
    auto_place_slot1: Any,
    containment: Any,
    controller: Any,
    decision_now_s: Any,
    decision_owner: Any,
    decision_source_max_age_s: Any,
    decision_source_timestamp_s: Any,
    estimator_config: Any,
    execute: Any,
    frame: Any,
    frame_source_monotonic_s: Any,
    motion_interlocks_ok: Any,
    placement_config: Any,
    placement_motion_active: Any,
    placement_sequencer: Any,
    root_config: Any,
    scene: Any,
    stationary: Any,
    vision_release_policy_enabled: Any,
    last_placement_output: Any,
    last_placement_runtime_diagnostics: Any,
    placement_alignment_ready_since_s: Any,
    placement_lowering_started: Any,
    placement_release_started: Any,
) -> _PlacementStep:
    """Advance the placement one frame: wait for alignment, lower, release, retreat.

    Called once per frame with the decision the base already made.  A wrong seating
    or a wrong retreat is diagnosed here; the postures themselves come from the
    slot configuration.
    """

    decision = None
    dispatch_result = None
    placement_output = None
    placement_runtime_diagnostics = None

    if execute:
        assert controller is not None
        dispatch_result = _dispatch_live_decision(
            controller,
            authority,
            decision_owner,
            decision,
            motion_interlocks_ok=motion_interlocks_ok,
            source_timestamp_s=decision_source_timestamp_s,
            source_max_age_s=decision_source_max_age_s,
        )

    placement_arrival_wait_reason: str | None = None
    placement_arrived_ready = bool(
        decision_owner is PalletControlOwner.FINE_SLOT1_SERVO
        and decision.state is PalletServoState.ARRIVED_HOLD
        and decision.arrived
    )
    placement_alignment_feature_ready = (
        _scene_has_metric_alignment_feature(
            scene,
            estimator_config.geometry,
        )
    )
    if placement_motion_active:
        placement_alignment_ready_since_s = None
        placement_alignment_dwell_ready = True
    elif placement_arrived_ready and placement_alignment_feature_ready:
        if placement_alignment_ready_since_s is None:
            placement_alignment_ready_since_s = decision_now_s
        placement_alignment_elapsed_s = (
            decision_now_s - placement_alignment_ready_since_s
        )
        placement_alignment_dwell_ready = (
            placement_alignment_elapsed_s
            >= placement_config.alignment_hold_before_place_s
        )
        if not placement_alignment_dwell_ready:
            placement_arrival_wait_reason = (
                "alignment_hold_before_place"
            )
    else:
        placement_alignment_ready_since_s = None
        placement_alignment_dwell_ready = False
        if placement_arrived_ready:
            placement_arrival_wait_reason = (
                "alignment_feature_unavailable"
            )
    placement_output: PlacementOutput | None = None
    placement_runtime_diagnostics: dict[str, Any] | None = None
    if (
        execute
        and auto_place_slot1
        and placement_sequencer is not None
        and (
            placement_motion_active
            or (
                placement_arrived_ready
                and placement_alignment_dwell_ready
            )
        )
        and dispatch_result
        in {"state_requires_persistent_zero", "exact_zero_decision"}
    ):
        assert controller is not None
        sample, placement_runtime_diagnostics = _placement_input(
            root_config,
            scene,
            controller,
            now_s=time.monotonic(),
            gap_observation_timestamp_s=frame_source_monotonic_s,
            gap_observation_sequence=frame.depth_frame_number,
            decision=decision,
            zero_acknowledged=_zero_command_acknowledged(controller),
            stationary=stationary,
            allow_vision_geometry_release=vision_release_policy_enabled,
        )
        placement_output = placement_sequencer.update(sample)
        fault_dispatch = _dispatch_placement_fault_hold_if_needed(
            controller,
            placement_output,
            lowering_started=placement_lowering_started,
            release_started=placement_release_started,
        )
        if fault_dispatch is not None:
            placement_runtime_diagnostics["runtime_request_dispatched"] = (
                fault_dispatch
            )
        elif (
            placement_output.request
            is PlacementRequest.LOWER_CARTESIAN_PLANNED
            and not placement_lowering_started
        ):
            descent_plan = placement_output.descent_plan
            if descent_plan is None or not descent_plan.valid:
                raise RuntimeError(
                    "planned Cartesian lowering requires a valid frozen "
                    "PlacementDescentPlan"
                )
            controller.start_cartesian_lowering_hold(
                descent_plan=descent_plan
            )
            placement_lowering_started = True
            assert containment is not None
            containment.mark_robot_touch()
            containment.mark_destination_commanded()
            placement_runtime_diagnostics["runtime_request_dispatched"] = (
                "start_cartesian_lowering_hold"
            )
        elif (
            placement_output.request is PlacementRequest.SPREAD_RELEASE
            and not placement_release_started
        ):
            controller.start_cartesian_release_hold()
            placement_release_started = True
            assert containment is not None
            containment.mark_robot_touch()
            containment.mark_destination_commanded()
            placement_runtime_diagnostics["runtime_request_dispatched"] = (
                "start_cartesian_release_hold"
            )
        else:
            placement_runtime_diagnostics["runtime_request_dispatched"] = (
                "none"
            )
        last_placement_output = placement_output
        last_placement_runtime_diagnostics = placement_runtime_diagnostics
        decision = _annotate_placement_output(
            decision,
            placement_output,
            placement_runtime_diagnostics,
        )
    elif (
        auto_place_slot1
        and placement_arrival_wait_reason is not None
        and placement_sequencer is not None
    ):
        diagnostics = dict(decision.diagnostics)
        diagnostics["placement_wait"] = {
            "reason": placement_arrival_wait_reason,
            "alignment_hold_before_place_s": (
                placement_config.alignment_hold_before_place_s
            ),
            "alignment_hold_elapsed_s": (
                0.0
                if placement_alignment_ready_since_s is None
                else max(
                    0.0,
                    decision_now_s
                    - placement_alignment_ready_since_s,
                )
            ),
            "metric_alignment_feature_ready": (
                placement_alignment_feature_ready
            ),
        }
        decision = PalletServoOutput(
            command=decision.command,
            state=decision.state,
            arrived=decision.arrived,
            hold_body=decision.hold_body,
            measurement_accepted=decision.measurement_accepted,
            reason=(
                f"{decision.reason}; "
                f"placement_wait:{placement_arrival_wait_reason}"
            ),
            diagnostics=diagnostics,
        )
    elif last_placement_output is not None:
        decision = _annotate_placement_output(
            decision,
            last_placement_output,
            last_placement_runtime_diagnostics,
        )

    return _PlacementStep(
        decision=decision,
        dispatch_result=dispatch_result,
        placement_output=placement_output,
        placement_runtime_diagnostics=placement_runtime_diagnostics,
        last_placement_output=last_placement_output,
        last_placement_runtime_diagnostics=last_placement_runtime_diagnostics,
        placement_alignment_ready_since_s=placement_alignment_ready_since_s,
        placement_lowering_started=placement_lowering_started,
        placement_release_started=placement_release_started,
    )

def run_pallet_live(
    root_config: Mapping[str, Any],
    *,
    execute: bool = False,
    auto_place_slot1: bool = False,
    ensure_slot1_ready: bool = False,
    slot: int | None = None,
    robot_address: str = "192.168.30.1:50051",
    robot_power: str = ".*",
    warmup_frames: int = 30,
    max_frames: int | None = None,
    headless: bool = False,
    window_name: str = "RB-Y1 Pallet Slot-1",
    output_mp4: str | Path | None = None,
    log_jsonl: str | Path | None = None,
    controller: _ControllerLike | None = None,
) -> int:
    """Run pallet perception, loaded-box slot-1 alignment, and gated placement.

    Execution is a standalone post-pick boundary: the previous process must be
    stopped, the configured loaded ready posture is verified, and this process
    becomes the sole combined body/mobility stream owner.
    """

    if not isinstance(root_config, Mapping):
        raise TypeError("root_config must be a mapping")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if execute and max_frames is not None:
        raise ValueError(
            "loaded execution cannot use max_frames because a bounded normal exit "
            "would abandon the non-daemon carried-load command owner"
        )
    if execute and not ensure_slot1_ready:
        raise ValueError("loaded slot-1 execution requires ensure_slot1_ready=True")
    if auto_place_slot1 and not execute:
        raise ValueError("slot-1 placement requires execute=True")

    # Before anything reaches the robot: a stranded carried load is far worse
    # than a rejected command line.
    _preflight_output_paths(log_jsonl, output_mp4)

    # Commissioning policy lives in the reviewed config, not on the command line.
    # Five acknowledgement flags never blocked an unsafe run; they only produced
    # runs that failed on a wrong flag combination.  The config still has to
    # enable each capability, so an unreviewed config cannot move the robot.
    # Resolve the slot before anything else prints or imports: an undemonstrated
    # slot is a configuration error, and a warning ahead of it only obscures it.
    pallet_block = _section(root_config, "pallet")
    selected_slot = int(
        pallet_block.get("default_slot", 1) if slot is None else slot
    )
    for member in ("hole_reference", "ready_pose_rad"):
        require_slot_member(root_config, selected_slot, member)
    if execute:
        for member in ("place_pose_deg", "retreat_pose_deg"):
            require_slot_member(root_config, selected_slot, member)

    calibration = _section(root_config, "calibration")
    if execute and not bool(calibration.get("absolute_base_validated", False)):
        # Not a blocker: the empirical registration is what every commissioning
        # run used.  The operator still has to see it named once per run.
        print(
            "warning: camera/base registration is nominal_unverified; base "
            "coordinates carry the empirical correction from "
            f"{calibration.get('base_translation_correction_source', 'an operator alignment')}",
            file=sys.stderr,
            flush=True,
        )
    grip_config = _section(root_config, "grip_interlock")
    geometry_only_policy_enabled = grip_config.get(
        "fixed_ready_geometry_only_commissioning_enabled", False
    )
    if not isinstance(geometry_only_policy_enabled, bool):
        raise ValueError(
            "grip_interlock.fixed_ready_geometry_only_commissioning_enabled must "
            "be a boolean"
        )
    if execute and not geometry_only_policy_enabled:
        raise RuntimeError(
            "loaded slot-1 motion requires "
            "grip_interlock.fixed_ready_geometry_only_commissioning_enabled=true "
            "in the reviewed config"
        )
    placement_section = _section(root_config, "placement")
    placement_config_enabled = bool(placement_section.get("enabled", False))
    vision_release_policy_enabled = bool(
        placement_section.get("vision_geometry_release_enabled", False)
    )
    if auto_place_slot1 and not placement_config_enabled:
        raise RuntimeError(
            "slot-1 placement requires placement.enabled=true in the reviewed config"
        )
    if auto_place_slot1 and not vision_release_policy_enabled:
        raise RuntimeError(
            "slot-1 placement requires "
            "placement.vision_geometry_release_enabled=true in the reviewed config"
        )
    if not execute and controller is not None:
        raise ValueError("controller is valid only with execute=True")

    if not headless and not _gui_available():
        raise RuntimeError(
            "no graphical display is available; use --headless with --output-mp4/"
            "--log-jsonl or run from the Jetson desktop session"
        )

    camera_config = _section(root_config, "camera")
    depth_config = _section(camera_config, "depth")
    color_config = _section(camera_config, "color")
    fps = int(depth_config.get("fps", 30))
    if fps != int(color_config.get("fps", -1)):
        raise ValueError("pallet live requires equal RGB and Depth FPS")
    stream_config = D435StreamConfig(
        depth_width=int(depth_config.get("width", 640)),
        depth_height=int(depth_config.get("height", 480)),
        color_width=int(color_config.get("width", 640)),
        color_height=int(color_config.get("height", 480)),
        fps=fps,
        align_color_to_depth=True,
        warmup_frames=int(warmup_frames),
    )

    # Imports stay below the standalone execute interlock.  In dry-run this is
    # still pure camera/perception code and cannot import rby1_sdk.
    from .pallet_geometry import PalletStackEstimator
    from .pallet_models import load_pallet_estimator_config

    estimator_config = load_pallet_estimator_config(root_config)
    estimator = PalletStackEstimator(estimator_config)
    slot1_hole_reference = load_slot1_hole_reference(root_config, selected_slot)
    acquisition_config = AcquisitionConfig.from_root_config(root_config)
    acquisition_servo = ForwardAcquireServo(acquisition_config)
    perception_config = _section(root_config, "perception")
    l_corner_gate = StationaryLCornerGate(
        acquisition_config.stationary_frames,
        metric_outer_size_m=estimator_config.geometry.outer_size_m,
        max_metric_center_spread_m=min(
            0.008,
            float(perception_config.get("live_center_spread_m", 0.008)),
        ),
    )
    hole_gate = StationaryHoleGate(
        required_frames=max(5, int(perception_config.get("stable_window_frames", 5))),
        minimum_duration_s=max(
            0.35,
            float(_section(root_config, "servo").get("arrival_min_duration_s", 0.35)),
        ),
        max_center_spread_m=min(
            0.008,
            float(perception_config.get("live_center_spread_m", 0.008)),
        ),
        max_yaw_spread_rad=min(
            math.radians(2.0),
            math.radians(float(perception_config.get("live_yaw_spread_deg", 2.0))),
        ),
    )
    servo = PalletSlot1Servo(PalletServoConfig.from_root_config(root_config))
    servo_bridge = ServoObservationBridge(
        maximum_prediction_age_s=_MAX_ODOMETRY_PREDICTION_AGE_S,
        odometry_freshness_s=acquisition_config.odometry_freshness_s,
    )
    placement_config = PlacementConfig.from_root_config(root_config)
    placement_sequencer = (
        Slot1PlacementSequencer(placement_config) if auto_place_slot1 else None
    )
    authority = CoarseFineAuthority()
    shutdown_pending = False
    # The alignment dwell used to depend on the first frame never reaching the
    # arrived branch; state it instead.
    placement_alignment_ready_since_s: float | None = None
    placement_lowering_started = False
    placement_release_started = False
    last_placement_output: PlacementOutput | None = None
    last_placement_runtime_diagnostics: dict[str, Any] | None = None
    scene_window: deque[dict[str, Any]] = deque(maxlen=30)
    calibration_status = "nominal_ready_assumed"
    T_base_depth = configured_T_base_from_depth(root_config)
    held_config = _section(root_config, "held_box")
    maximum_box_height_m = float(held_config.get("maximum_height_m", 0.164))
    box_bottom_uncertainty_m = float(
        held_config.get("fixed_ready_box_bottom_uncertainty_m", 0.015)
    )
    if not math.isfinite(maximum_box_height_m) or maximum_box_height_m <= 0.0:
        raise ValueError("held_box.maximum_height_m must be finite and positive")
    if (
        not math.isfinite(box_bottom_uncertainty_m)
        or box_bottom_uncertainty_m < 0.0
    ):
        raise ValueError(
            "held_box.fixed_ready_box_bottom_uncertainty_m must be finite and "
            "non-negative"
        )
    held_proxy = _nominal_held_pose(root_config)

    video_writer: Any | None = None
    log_stream: TextIO | None = None
    window_created = False
    frame_count = 0
    accepted_scene_sequence = 0
    frame_gate = LiveFrameGate(
        maximum_capture_age_s=float(camera_config.get("frame_fresh_after_s", 0.20)),
        maximum_rgb_depth_timestamp_skew_s=float(
            camera_config.get("maximum_rgb_depth_timestamp_skew_s", 0.05)
        ),
    )
    containment: ActuationContainmentState | None = None
    try:
        if ensure_slot1_ready and controller is None:
            from .pallet_ready import ensure_slot1_ready_from_config

            ensure_slot1_ready_from_config(
                root_config,
                address=robot_address,
                power=robot_power,
                prepare_for_stream=execute,
            )
            calibration_status = "nominal_unverified_ready_posture_checked_at_start"
        if execute:
            if controller is None:
                from .pallet_control import PalletControlConfig, RBY1PalletController

                controller = RBY1PalletController(
                    execute=True,
                    config=PalletControlConfig.from_root_config(
                        root_config,
                        address_override=robot_address,
                        slot=selected_slot,
                    ),
                )
            containment = ActuationContainmentState(controller)
            _prepare_loaded_ready_actuation(
                controller,
                float(
                    _section(root_config, "robot").get(
                        "ready_transition_minimum_time_s", 5.0
                    )
                ),
                containment,
            )
            T_base_depth = measured_T_base_from_depth(
                root_config, controller.get_measured_T_base_head()
            )
            right_eef, left_eef = controller.get_measured_eef_transforms()
            held_proxy = _fixed_ready_held_pose(root_config, right_eef, left_eef)
            calibration_status = "nominal_unverified_operator_accepted"
            print(
                "warning: loaded slot-1 MVP is using fixed-ready FK/EEF geometry "
                "and depth clearance evidence; placement does not read F/T",
                file=sys.stderr,
                flush=True,
            )

        log_stream = _open_log(None if log_jsonl is None else Path(log_jsonl))
        with RealSenseAdapter(stream_config) as camera:
            contract = validate_live_camera_profile(
                camera.active_profile_metadata(), root_config
            )
            if not headless:
                import cv2  # type: ignore[import-not-found]

                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                window_created = True

            while max_frames is None or frame_count < max_frames:
                try:
                    frame = camera.capture()
                except KeyboardInterrupt:
                    if not execute:
                        break
                    assert containment is not None
                    should_exit = containment.request_shutdown_hold(
                        next_owner="operator-successor-required"
                    )
                    if should_exit:
                        print(
                            "DANGER: forced cancellation followed the explicit second "
                            "interrupt; squeeze/body continuity is no longer guaranteed",
                            file=sys.stderr,
                        )
                        break
                    was_fine = authority.owner is PalletControlOwner.FINE_SLOT1_SERVO
                    authority.request_shutdown_hold()
                    shutdown_pending = True
                    if was_fine:
                        servo.request_shutdown_hold(time.monotonic())
                    print(
                        "warning: first interrupt entered SHUTDOWN_PENDING_HOLD; "
                        "body hold remains active and a successor acknowledgement is "
                        "required. Interrupt again only for unsafe forced cancellation.",
                        file=sys.stderr,
                    )
                    continue

                frame_received_monotonic_s = time.monotonic()
                estimator_started_monotonic_s = frame_received_monotonic_s
                validated_frame_age_s = frame_gate.validate(frame)
                frame_source_monotonic_s = (
                    frame_received_monotonic_s - validated_frame_age_s
                )

                observed = observe_pallet_frame(
                    frame,
                    root_config=root_config,
                    contract=contract,
                    estimator=estimator,
                    controller=controller if execute else None,
                    calibration_status=calibration_status,
                    configured_T_base_depth=T_base_depth,
                    configured_held_proxy=held_proxy,
                    capture_monotonic_s=frame_source_monotonic_s,
                    accepted_scene_sequence=accepted_scene_sequence,
                    maximum_box_height_m=maximum_box_height_m,
                    box_bottom_uncertainty_m=box_bottom_uncertainty_m,
                )
                T_base_depth = observed.T_base_depth
                held_proxy = observed.held_proxy
                scene = observed.scene
                estimator_finished_monotonic_s = observed.estimator_finished_s
                decision_now_s = observed.decision_now_s
                frame_result_fresh = observed.result_fresh
                fresh_scene_sample = observed.scene_sample
                accepted_scene_sequence = observed.accepted_scene_sequence
                # The overlay draws on the same colour image the estimator used.
                color = (
                    frame.color_on_depth_bgr
                    if frame.color_on_depth_bgr is not None
                    else frame.raw_color_bgr
                )
                _update_scene_window(
                    scene_window,
                    frame_result_fresh=frame_result_fresh,
                    fresh_sample=fresh_scene_sample,
                )
                grip_result: Any | None = None
                base_motion = _decide_base_motion(
                    acquisition_config=acquisition_config,
                    acquisition_servo=acquisition_servo,
                    authority=authority,
                    auto_place_slot1=auto_place_slot1,
                    controller=controller,
                    decision_now_s=decision_now_s,
                    estimator_config=estimator_config,
                    execute=execute,
                    frame_result_fresh=frame_result_fresh,
                    frame_source_monotonic_s=frame_source_monotonic_s,
                    geometry_only_policy_enabled=geometry_only_policy_enabled,
                    hole_gate=hole_gate,
                    l_corner_gate=l_corner_gate,
                    placement_lowering_started=placement_lowering_started,
                    placement_release_started=placement_release_started,
                    placement_sequencer=placement_sequencer,
                    scene=scene,
                    scene_window=scene_window,
                    servo=servo,
                    servo_bridge=servo_bridge,
                    shutdown_pending=shutdown_pending,
                    slot1_hole_reference=slot1_hole_reference,
                )
                acquisition_output = base_motion.acquisition_output
                bridge_diagnostics = base_motion.bridge_diagnostics
                decision = base_motion.decision
                decision_owner = base_motion.decision_owner
                decision_source_max_age_s = base_motion.decision_source_max_age_s
                decision_source_timestamp_s = base_motion.decision_source_timestamp_s
                dispatch_result = base_motion.dispatch_result
                grip_result = base_motion.grip_result
                hole_status = base_motion.hole_status
                l_status = base_motion.l_status
                motion_interlock_reason = base_motion.motion_interlock_reason
                motion_interlocks_ok = base_motion.motion_interlocks_ok
                odometry = base_motion.odometry
                odometry_error = base_motion.odometry_error
                placement_motion_active = base_motion.placement_motion_active
                stationary = base_motion.stationary
                stationary_source = base_motion.stationary_source
                placement_step = _advance_placement(
                    authority=authority,
                    auto_place_slot1=auto_place_slot1,
                    containment=containment,
                    controller=controller,
                    decision_now_s=decision_now_s,
                    decision_owner=decision_owner,
                    decision_source_max_age_s=decision_source_max_age_s,
                    decision_source_timestamp_s=decision_source_timestamp_s,
                    estimator_config=estimator_config,
                    execute=execute,
                    frame=frame,
                    frame_source_monotonic_s=frame_source_monotonic_s,
                    motion_interlocks_ok=motion_interlocks_ok,
                    placement_config=placement_config,
                    placement_motion_active=placement_motion_active,
                    placement_sequencer=placement_sequencer,
                    root_config=root_config,
                    scene=scene,
                    stationary=stationary,
                    vision_release_policy_enabled=vision_release_policy_enabled,
                    last_placement_output=last_placement_output,
                    last_placement_runtime_diagnostics=last_placement_runtime_diagnostics,
                    placement_alignment_ready_since_s=placement_alignment_ready_since_s,
                    placement_lowering_started=placement_lowering_started,
                    placement_release_started=placement_release_started,
                )
                decision = placement_step.decision
                dispatch_result = placement_step.dispatch_result
                placement_output = placement_step.placement_output
                placement_runtime_diagnostics = placement_step.placement_runtime_diagnostics
                last_placement_output = placement_step.last_placement_output
                last_placement_runtime_diagnostics = placement_step.last_placement_runtime_diagnostics
                placement_alignment_ready_since_s = placement_step.placement_alignment_ready_since_s
                placement_lowering_started = placement_step.placement_lowering_started
                placement_release_started = placement_step.placement_release_started
                loop_finished_monotonic_s = time.monotonic()
                loop_timing = {
                    "capture_age_s": float(validated_frame_age_s),
                    "estimator_ms": 1000.0
                    * (
                        estimator_finished_monotonic_s
                        - estimator_started_monotonic_s
                    ),
                    "post_estimator_decision_ms": 1000.0
                    * (loop_finished_monotonic_s - estimator_finished_monotonic_s),
                    "loop_ms": 1000.0
                    * (loop_finished_monotonic_s - frame_received_monotonic_s),
                }

                overlay = draw_live_overlay(
                    color,
                    scene,
                    estimator.last_evidence,
                    held_proxy,
                    decision,
                    T_base_depth,
                    contract.depth_intrinsics,
                    slot1_hole_reference,
                    estimator_config.geometry,
                    execute=execute,
                    acquisition=acquisition_output,
                    l_gate=l_status,
                    hole_gate=hole_status,
                    stationary_source=stationary_source,
                    motion_interlock_reason=motion_interlock_reason,
                    dispatch_result=dispatch_result,
                    placement=placement_output or last_placement_output,
                    placement_runtime=(
                        placement_runtime_diagnostics
                        or last_placement_runtime_diagnostics
                    ),
                )
                if video_writer is None and output_mp4 is not None:
                    video_writer = _open_video(Path(output_mp4), overlay.shape[:2], fps)
                if video_writer is not None:
                    video_writer.write(overlay)
                record = _telemetry_record(
                    frame_count,
                    float(frame.hardware_timestamp_ms or frame.depth_timestamp_ms),
                    scene,
                    held_proxy,
                    decision,
                    execute=execute,
                    controller=controller,
                    acquisition=acquisition_output,
                    l_gate=l_status,
                    hole_gate=hole_status,
                    stationary_source=stationary_source,
                    odometry=odometry,
                    odometry_error=odometry_error,
                    motion_interlocks_ok=motion_interlocks_ok,
                    motion_interlock_reason=motion_interlock_reason,
                    grip_result=grip_result,
                    dispatch_result=dispatch_result,
                    T_base_depth=T_base_depth,
                    slot1_hole_reference=slot1_hole_reference,
                    placement=placement_output or last_placement_output,
                    placement_runtime_diagnostics=(
                        placement_runtime_diagnostics
                        or last_placement_runtime_diagnostics
                    ),
                    loop_timing=loop_timing,
                    geometry=estimator_config.geometry,
                    estimator_config=estimator_config,
                    bridge_diagnostics=bridge_diagnostics,
                )
                _write_record(log_stream, record)
                if frame_count % fps == 0:
                    placement_suffix = (
                        ""
                        if (placement_output or last_placement_output) is None
                        else (
                            " placement="
                            f"{(placement_output or last_placement_output).state.value}:"
                            f"{(placement_output or last_placement_output).reason}"
                        )
                    )
                    feature_source = decision.diagnostics.get(
                        "observed_feature_source",
                        "complete_hole" if scene.valid else "unavailable",
                    )
                    print(
                        f"[pallet] frame={frame_count} vision={feature_source} "
                        f"state={decision.state.value} reason={decision.reason} "
                        f"dispatch={dispatch_result} interlock="
                        f"{motion_interlock_reason or 'PASS'}"
                        f"{placement_suffix}",
                        file=sys.stderr,
                    )
                frame_count += 1

                if not headless:
                    import cv2  # type: ignore[import-not-found]

                    cv2.imshow(window_name, overlay)
                    key = int(cv2.waitKey(1)) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):
                        if not execute:
                            break
                        assert containment is not None
                        should_exit = containment.request_shutdown_hold(
                            next_owner="operator-successor-required"
                        )
                        if not should_exit:
                            was_fine = (
                                authority.owner is PalletControlOwner.FINE_SLOT1_SERVO
                            )
                            authority.request_shutdown_hold()
                            shutdown_pending = True
                            if was_fine:
                                servo.request_shutdown_hold(time.monotonic())
                            print(
                                "warning: shutdown is pending; press q again only for "
                                "unsafe best-effort cancellation without successor",
                                file=sys.stderr,
                            )
                        else:
                            print(
                                "DANGER: forced cancel; body/grip continuity is no longer guaranteed",
                                file=sys.stderr,
                            )
                            break
    except BaseException:
        if execute and containment is not None and containment.robot_touched:
            containment.confirm_persistent_support()
            containment.block_until_escape_is_safe()
        elif execute and controller is not None and controller.is_connected:
            # Connection/bootstrap failures before the first command own no
            # carried-load stream and may close normally.  Do not leave the
            # measured-state update thread alive on a rejected ready posture.
            try:
                controller.close()
            except BaseException as close_error:
                print(
                    "warning: pre-command pallet controller cleanup failed: "
                    f"{type(close_error).__name__}:{close_error}",
                    file=sys.stderr,
                    flush=True,
                )
        raise
    finally:
        if video_writer is not None:
            video_writer.release()
        if log_stream is not None:
            log_stream.close()
        if window_created:
            try:
                import cv2  # type: ignore[import-not-found]

                cv2.destroyWindow(window_name)
            except Exception:
                pass
        # Never imply that resource cleanup preserves a loaded robot.  The
        # execute path remains open unless explicit forced cancellation or a
        # separately acknowledged owner handoff closes it.
    return 0


__all__ = [
    "ActuationContainmentState",
    "HeldPoseProxy",
    "LiveFrameGate",
    "LiveCameraContract",
    "ServoObservationBridge",
    "configured_T_base_from_depth",
    "measured_T_base_from_depth",
    "run_pallet_live",
    "validate_live_camera_profile",
]
