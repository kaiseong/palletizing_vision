"""Software-only Phase-2 refusal and arrival matrix for horizontal picking.

These tests stop at the pure ``MobileVisualServo`` boundary.  The dependent
event trace models the only condition under which ``AutoGrabRuntime`` may begin
its stop/release and grasp handoff; it never constructs a robot, command
stream, SDK object, or camera.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import math
from typing import Callable, Iterator

import pytest

from parcel_pose_common.mobile_servo import (
    MobileVisualServo,
    PoseMeasurement,
    ServoConfig,
    ServoDecision,
    ServoState,
)


@pytest.fixture(autouse=True)
def _hardware_imports_are_forbidden(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Turn any accidental lazy hardware import into an immediate test failure."""

    original_import_module = importlib.import_module
    attempted_hardware_imports: list[str] = []

    def guarded_import_module(name: str, package: str | None = None) -> object:
        root = name.lstrip(".").split(".", maxsplit=1)[0]
        if root in {"rby1_sdk", "pyrealsense2"}:
            attempted_hardware_imports.append(name)
            pytest.fail(f"software-only servo test attempted hardware import: {name}")
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", guarded_import_module)
    yield
    assert attempted_hardware_imports == []


@dataclass
class _FakeClock:
    now_s: float = 0.0

    def set(self, now_s: float) -> float:
        assert now_s >= self.now_s
        self.now_s = now_s
        return self.now_s

    def __call__(self) -> float:
        return self.now_s


@dataclass
class _DependentEffectTrace:
    """Count downstream effects that a handoff-ready decision would authorize."""

    handoff_calls: int = 0
    grasp_calls: int = 0
    events: list[str] = field(default_factory=list)

    def observe(self, decision: ServoDecision) -> None:
        if decision.handoff_ready:
            self.handoff_calls += 1
            self.events.append("handoff")
            # A trace event only: no manipulator implementation is called here.
            self.grasp_calls += 1
            self.events.append("grasp")


@dataclass
class _SoftwarePickingHarness:
    config: ServoConfig = field(default_factory=ServoConfig)
    clock: _FakeClock = field(default_factory=_FakeClock)
    effects: _DependentEffectTrace = field(default_factory=_DependentEffectTrace)
    decisions: list[ServoDecision] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.servo = MobileVisualServo(self.config)
        self.decisions.append(self.servo.start(self.clock()))

    def step(
        self,
        measurement: PoseMeasurement | None,
        *,
        at_s: float,
    ) -> ServoDecision:
        decision = self.servo.step(measurement, now_s=self.clock.set(at_s))
        self.decisions.append(decision)
        self.effects.observe(decision)
        return decision


def _pose(
    *,
    x_m: float = 0.740,
    y_m: float = 0.0,
    yaw_rad: float = math.pi / 2.0,
    timestamp_s: float,
) -> PoseMeasurement:
    return PoseMeasurement(
        (x_m, y_m),
        timestamp_s=timestamp_s,
        long_axis_yaw_rad=yaw_rad,
    )


def _prime_tracking(harness: _SoftwarePickingHarness) -> ServoDecision:
    """Build the default three-sample filter and produce a non-zero command."""

    decisions = [
        harness.step(_pose(x_m=0.760, timestamp_s=now_s), at_s=now_s)
        for now_s in (0.10, 0.20, 0.30)
    ]
    assert [decision.state for decision in decisions] == [
        ServoState.ACQUIRING,
        ServoState.ACQUIRING,
        ServoState.TRACKING,
    ]
    assert not decisions[-1].command.is_zero
    return decisions[-1]


def _assert_refused_without_dependent_effects(
    harness: _SoftwarePickingHarness,
    decision: ServoDecision,
) -> None:
    assert decision.command.is_zero
    assert decision.handoff_ready is False
    assert harness.effects.handoff_calls == 0
    assert harness.effects.grasp_calls == 0
    assert harness.effects.events == []


def test_matrix_pins_current_horizontal_servo_defaults() -> None:
    config = ServoConfig()

    assert config.target_xy_m == (0.740, 0.0)
    assert config.target_long_axis_yaw_rad == pytest.approx(math.pi / 2.0)
    assert config.filter_window == 3
    assert config.stale_after_s == pytest.approx(0.30)
    assert config.jump_threshold_m == pytest.approx(0.030)
    assert config.yaw_jump_threshold_rad == pytest.approx(math.radians(15.0))
    assert config.jump_reseed_frames == 3
    assert config.arrival_inner_m == pytest.approx(0.010)
    assert config.arrival_outer_m == pytest.approx(0.015)
    assert config.arrival_yaw_inner_rad == pytest.approx(math.radians(3.0))
    assert config.arrival_yaw_outer_rad == pytest.approx(math.radians(5.0))
    assert config.arrival_min_frames == 3
    assert config.arrival_min_duration_s == pytest.approx(0.35)
    assert config.lost_abort_after_s == pytest.approx(2.0)
    assert config.timeout_s == pytest.approx(30.0)


@pytest.mark.parametrize(
    ("measurement_factory", "reason"),
    [
        (lambda now_s: None, "pose_missing"),
        (PoseMeasurement.invalid, "pose_invalid"),
    ],
)
def test_missing_or_invalid_pose_immediately_zeros_motion_and_never_hands_off(
    measurement_factory: Callable[[float], PoseMeasurement | None],
    reason: str,
) -> None:
    harness = _SoftwarePickingHarness()
    _prime_tracking(harness)

    decision = harness.step(measurement_factory(0.31), at_s=0.31)

    assert decision.state is ServoState.POSE_LOST
    assert decision.reason == reason
    _assert_refused_without_dependent_effects(harness, decision)

    lost_timeout = harness.step(measurement_factory(2.32), at_s=2.32)
    assert lost_timeout.state is ServoState.ABORTED
    assert lost_timeout.reason == "pose_lost_timeout"
    _assert_refused_without_dependent_effects(harness, lost_timeout)


def test_stale_measurement_immediately_zeros_motion_and_never_hands_off() -> None:
    harness = _SoftwarePickingHarness()
    _prime_tracking(harness)

    stale = harness.step(
        _pose(x_m=0.760, timestamp_s=0.01),
        at_s=0.32,
    )

    assert stale.state is ServoState.POSE_LOST
    assert stale.reason == "pose_stale"
    assert 0.32 - 0.01 > harness.config.stale_after_s
    _assert_refused_without_dependent_effects(harness, stale)


@pytest.mark.parametrize(
    ("changed_pose", "jump_kind"),
    [
        ({"x_m": 0.791}, "x"),
        ({"x_m": 0.760, "y_m": 0.031}, "y"),
        (
            {"x_m": 0.760, "yaw_rad": math.pi / 2.0 + math.radians(16.0)},
            "yaw",
        ),
    ],
)
def test_x_y_or_yaw_jump_is_rejected_with_zero_and_no_handoff(
    changed_pose: dict[str, float],
    jump_kind: str,
) -> None:
    harness = _SoftwarePickingHarness()
    _prime_tracking(harness)

    jumped = harness.step(_pose(timestamp_s=0.31, **changed_pose), at_s=0.31)

    assert jump_kind in {"x", "y", "yaw"}
    assert jumped.state is ServoState.POSE_LOST
    assert jumped.reason == "jump_rejected"
    assert jumped.measurement_accepted is False
    _assert_refused_without_dependent_effects(harness, jumped)


def test_global_timeout_preempts_even_a_fresh_valid_pose() -> None:
    harness = _SoftwarePickingHarness()
    _prime_tracking(harness)

    timeout = harness.step(
        _pose(x_m=0.760, timestamp_s=harness.config.timeout_s),
        at_s=harness.config.timeout_s,
    )

    assert timeout.state is ServoState.ABORTED
    assert timeout.reason == "timeout"
    _assert_refused_without_dependent_effects(harness, timeout)


def test_xy_arrival_is_radial_not_two_independent_axis_gates() -> None:
    harness = _SoftwarePickingHarness()
    axis_error_m = 0.008
    decisions = [
        harness.step(
            _pose(
                x_m=harness.config.target_xy_m[0] + axis_error_m,
                y_m=harness.config.target_xy_m[1] + axis_error_m,
                timestamp_s=now_s,
            ),
            at_s=now_s,
        )
        for now_s in (0.10, 0.20, 0.30)
    ]

    decision = decisions[-1]
    assert axis_error_m < harness.config.arrival_inner_m
    assert math.hypot(axis_error_m, axis_error_m) > harness.config.arrival_inner_m
    assert decision.state is ServoState.TRACKING
    assert decision.handoff_ready is False
    assert not decision.command.is_zero
    assert harness.effects.handoff_calls == 0
    assert harness.effects.grasp_calls == 0


def test_modulo_pi_yaw_and_outer_hysteresis_require_full_default_dwell() -> None:
    harness = _SoftwarePickingHarness()
    target_x, target_y = harness.config.target_xy_m

    acquiring_and_hold = [
        harness.step(
            _pose(
                x_m=target_x,
                y_m=target_y,
                yaw_rad=yaw_rad,
                timestamp_s=now_s,
            ),
            at_s=now_s,
        )
        for now_s, yaw_rad in (
            (0.10, -math.pi / 2.0),
            (0.20, math.pi / 2.0),
            (0.30, -math.pi / 2.0),
        )
    ]
    assert [decision.state for decision in acquiring_and_hold] == [
        ServoState.ACQUIRING,
        ServoState.ACQUIRING,
        ServoState.HOLDING,
    ]
    assert acquiring_and_hold[-1].yaw_error_rad == pytest.approx(0.0)

    outer_band_decisions = [
        harness.step(
            _pose(
                x_m=target_x + 0.012,
                y_m=target_y,
                yaw_rad=math.pi / 2.0 + math.radians(4.0),
                timestamp_s=now_s,
            ),
            at_s=now_s,
        )
        for now_s in (0.40, 0.50)
    ]
    assert all(
        decision.state is ServoState.HOLDING
        and decision.command.is_zero
        and not decision.handoff_ready
        for decision in outer_band_decisions
    )
    assert harness.effects.handoff_calls == 0
    assert harness.effects.grasp_calls == 0

    arrived = harness.step(
        _pose(
            x_m=target_x + 0.012,
            y_m=target_y,
            yaw_rad=-math.pi / 2.0 + math.radians(4.0),
            timestamp_s=0.66,
        ),
        at_s=0.66,
    )
    assert 0.66 - 0.30 > harness.config.arrival_min_duration_s
    assert arrived.state is ServoState.ARRIVED
    assert arrived.reason == "arrival_stable"
    assert arrived.command.is_zero
    assert arrived.handoff_ready is True
    assert harness.effects.handoff_calls == 1
    assert harness.effects.grasp_calls == 1

    after_rising_edge = harness.step(None, at_s=0.70)
    assert after_rising_edge.state is ServoState.ARRIVED
    assert after_rising_edge.command.is_zero
    assert after_rising_edge.handoff_ready is False
    assert harness.effects.handoff_calls == 1
    assert harness.effects.grasp_calls == 1


def test_crossing_outer_hysteresis_resets_dwell_before_handoff() -> None:
    harness = _SoftwarePickingHarness()
    target_x, target_y = harness.config.target_xy_m

    for now_s in (0.10, 0.20, 0.30):
        decision = harness.step(_pose(timestamp_s=now_s), at_s=now_s)
    assert decision.state is ServoState.HOLDING

    still_holding = harness.step(
        _pose(
            x_m=target_x + 0.012,
            y_m=target_y,
            yaw_rad=math.pi / 2.0 + math.radians(4.0),
            timestamp_s=0.40,
        ),
        at_s=0.40,
    )
    assert still_holding.state is ServoState.HOLDING

    outside_outer = harness.step(
        _pose(
            x_m=target_x + 0.016,
            y_m=target_y,
            yaw_rad=math.pi / 2.0 + math.radians(6.0),
            timestamp_s=0.50,
        ),
        at_s=0.50,
    )
    assert outside_outer.state is ServoState.TRACKING
    assert outside_outer.handoff_ready is False
    assert harness.effects.handoff_calls == 0
    assert harness.effects.grasp_calls == 0

    # Returning to the inner gate does not inherit the earlier dwell.
    for now_s in (0.60, 0.70):
        not_yet = harness.step(_pose(timestamp_s=now_s), at_s=now_s)
        assert not_yet.handoff_ready is False
    restarted_hold = harness.step(_pose(timestamp_s=0.80), at_s=0.80)
    assert restarted_hold.state is ServoState.HOLDING
    assert restarted_hold.handoff_ready is False

    before_new_dwell = harness.step(_pose(timestamp_s=1.00), at_s=1.00)
    assert before_new_dwell.state is ServoState.HOLDING
    assert before_new_dwell.handoff_ready is False
    arrived = harness.step(_pose(timestamp_s=1.16), at_s=1.16)
    assert arrived.state is ServoState.ARRIVED
    assert arrived.handoff_ready is True
    assert harness.effects.handoff_calls == 1
    assert harness.effects.grasp_calls == 1
