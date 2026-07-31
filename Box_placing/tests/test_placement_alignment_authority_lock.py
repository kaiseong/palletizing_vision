"""Pure Phase-4 tests for the alignment-to-placement authority lock."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from parcel_pose_placing.pallet_acquisition import (
    CoarseFineAuthority,
    PalletControlOwner,
)
from parcel_pose_placing.placing_session import PlacingSession


def _command(
    vx_mps: float = 0.0,
    vy_mps: float = 0.0,
    wz_radps: float = 0.0,
) -> SimpleNamespace:
    return SimpleNamespace(vx_mps=vx_mps, vy_mps=vy_mps, wz_radps=wz_radps)


def _handoff_to_fine(authority: CoarseFineAuthority) -> None:
    authority.handoff_to_fine(
        SimpleNamespace(fine_handoff_requested=True, is_exact_zero=True),
        zero_command_acknowledged=True,
        wheel_stopped=True,
    )


def test_early_coarse_release_is_refused_without_locking() -> None:
    authority = CoarseFineAuthority()

    with pytest.raises(
        RuntimeError,
        match="requires fine slot-1 servo ownership",
    ):
        authority.release_alignment_for_placement()

    assert authority.owner is PalletControlOwner.FORWARD_ACQUISITION
    assert authority.placement_locked is False


def test_fine_handoff_can_release_alignment_once_and_remain_fine_owned() -> None:
    authority = CoarseFineAuthority()
    _handoff_to_fine(authority)

    assert authority.release_alignment_for_placement() is True
    assert authority.release_alignment_for_placement() is True
    assert authority.owner is PalletControlOwner.FINE_SLOT1_SERVO
    assert authority.placement_locked is True


@pytest.mark.parametrize(
    "command",
    (
        _command(vx_mps=0.001),
        _command(vy_mps=-0.001),
        _command(wz_radps=0.001),
    ),
)
def test_released_alignment_allows_exact_zero_only(command: SimpleNamespace) -> None:
    authority = CoarseFineAuthority()
    _handoff_to_fine(authority)
    authority.release_alignment_for_placement()

    authority.assert_publish(PalletControlOwner.FINE_SLOT1_SERVO, _command())
    with pytest.raises(
        RuntimeError,
        match="released placement alignment permits exact zero only",
    ):
        authority.assert_publish(PalletControlOwner.FINE_SLOT1_SERVO, command)


def test_shutdown_hold_also_locks_to_exact_zero() -> None:
    authority = CoarseFineAuthority()
    authority.request_shutdown_hold()

    assert authority.placement_locked is True
    authority.assert_publish(PalletControlOwner.SHUTDOWN_HOLD, _command())
    with pytest.raises(
        RuntimeError,
        match="released placement alignment permits exact zero only",
    ):
        authority.assert_publish(
            PalletControlOwner.SHUTDOWN_HOLD,
            _command(vx_mps=0.001),
        )


def test_placing_session_uses_the_placement_release_lock_not_shutdown() -> None:
    events: list[str] = []

    class Authority:
        def release_alignment_for_placement(self) -> bool:
            events.append("placement_release")
            return True

        def request_shutdown_hold(self) -> None:  # pragma: no cover - forbidden path
            raise AssertionError("placing release must not request shutdown ownership")

    session = object.__new__(PlacingSession)
    session._alignment_released = False
    session._pending_descent_plan = object()
    session._shutdown_pending = False
    session._stack = SimpleNamespace(authority=Authority())

    assert session.release_alignment() is True
    assert session.release_alignment() is True
    assert events == ["placement_release"]
    assert session._shutdown_pending is False
