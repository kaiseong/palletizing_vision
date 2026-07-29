"""Containment must stay unbounded when recoverable and bounded when not."""

from __future__ import annotations

import itertools

import pytest

from parcel_pose.pallet_runtime import ActuationContainmentState


class FakeTelemetry:
    def __init__(self, phase: str = "FAULT_HOLD") -> None:
        self.phase = type("Phase", (), {"value": phase})()
        self.zero_latched = True
        self.body_hold_included = True
        self.mobility_included = True
        self.last_sent_mobility = type("Mobility", (), {"is_zero": True})()


class FakeOffer:
    acknowledged = False


class ExpiredStreamController:
    """Every hold attempt fails the way an expired RB-Y1 stream does."""

    def __init__(self) -> None:
        self.close_calls: list[bool] = []

    def ensure_persistent_zero_body_hold(self) -> None:
        raise RuntimeError(
            "fault zero/body-hold pump failed: This command stream is expired"
        )

    def telemetry(self) -> FakeTelemetry:
        return FakeTelemetry()

    def transfer_owner(self, next_owner: str) -> FakeOffer:
        return FakeOffer()

    def close(self, *, force: bool = False) -> bool:
        self.close_calls.append(force)
        return True


class RecoverableController(ExpiredStreamController):
    def ensure_persistent_zero_body_hold(self) -> None:
        return None


def fake_clock(step_s: float):
    counter = itertools.count()
    return lambda: step_s * next(counter)


def containment(controller, *, timeout_s: float, step_s: float):
    state = ActuationContainmentState(
        controller,  # type: ignore[arg-type]
        unrecoverable_timeout_s=timeout_s,
    )
    state._clock = fake_clock(step_s)
    state.mark_robot_touch()
    state.mark_destination_commanded()
    state.mark_destination_steady()
    return state


def test_expired_stream_is_classified_as_unrecoverable() -> None:
    state = containment(ExpiredStreamController(), timeout_s=1.0, step_s=0.5)
    assert state.confirm_persistent_support() is False
    assert state.hold_unrecoverable
    assert "expired" in (state.last_hold_error or "")


def test_recoverable_hold_is_never_marked_unrecoverable() -> None:
    state = containment(RecoverableController(), timeout_s=1.0, step_s=0.5)
    assert state.confirm_persistent_support() is True
    assert not state.hold_unrecoverable
    assert state.last_hold_error is None


def test_escape_deadline_only_fires_for_unrecoverable_holds() -> None:
    state = containment(RecoverableController(), timeout_s=1.0, step_s=10.0)
    state.confirm_persistent_support()
    assert state.escape_deadline_reached() is False


def test_unrecoverable_containment_returns_within_the_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("parcel_pose.pallet_runtime.time.sleep", lambda _s: None)
    controller = ExpiredStreamController()
    state = containment(controller, timeout_s=1.0, step_s=0.4)
    # Would otherwise loop forever: no successor can ever acknowledge.
    state.block_until_escape_is_safe()
    assert state.hold_unrecoverable
    assert not state.successor_acknowledged
    assert not state.forced_cancel
    # Containment reports instead of silently force-cancelling the stream.
    assert controller.close_calls == []


def test_recovered_hold_clears_the_unrecoverable_latch() -> None:
    controller = ExpiredStreamController()
    state = containment(controller, timeout_s=5.0, step_s=0.1)
    assert state.confirm_persistent_support() is False
    assert state.hold_unrecoverable
    controller.ensure_persistent_zero_body_hold = lambda: None  # type: ignore[method-assign]
    assert state.confirm_persistent_support() is True
    assert not state.hold_unrecoverable
    assert state.unrecoverable_elapsed_s() == 0.0


def test_untouched_robot_never_blocks() -> None:
    state = ActuationContainmentState(ExpiredStreamController())  # type: ignore[arg-type]
    assert state.confirm_persistent_support() is True
    assert state.support_owner == "not_touched"
