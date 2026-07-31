"""Software-only proof of the slot-1 placement lifecycle facts.

The fakes below model only the already existing controller boundary.  They do
not import the SDK, open a camera, create a robot, or send a hardware command.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from parcel_pose_placing.placement_lifecycle import (
    PlaceAcknowledgedAndReleased,
    PlaceAlignmentStoppedAndReleased,
    PlacementLifecycleError,
    PlacementLifecycleRuntime,
    RetreatCompleted,
)


@dataclass
class _Controller:
    fail_at: str | None = None
    zero_latched: bool = True
    wheel_stopped: bool = True
    zero_acknowledged: bool = True
    place_acknowledged: bool = True
    retreat_acknowledged: bool = True
    close_failures_remaining: int = 0
    close_false_remaining: int = 0
    events: list[str] = field(default_factory=list)
    phase: str = "alignment"

    def _step(self, name: str) -> None:
        self.events.append(name)
        if self.fail_at == name:
            raise RuntimeError(f"forced {name} failure")

    def send_zero_mobility_hold(self, *, latch: bool = True) -> None:
        assert latch is True
        self._step("exact_zero_latch")
        self.phase = "zero"

    def wait_for_wheel_stop(self, timeout_s: float) -> Any:
        assert timeout_s > 0.0
        self._step("measured_wheel_stop")
        return SimpleNamespace(stopped=self.wheel_stopped)

    def start_cartesian_lowering_hold(self, *, descent_plan: Any) -> object:
        assert descent_plan is _DESCENT_PLAN
        self._step("place_command")
        self.phase = "place"
        return object()

    def start_cartesian_release_hold(self) -> object:
        self._step("retreat_command")
        self.phase = "retreat"
        return object()

    def placement_telemetry(self) -> Any:
        if self.phase == "zero":
            self._step("exact_zero_ack")
            acknowledged = self.zero_acknowledged
        elif self.phase == "place":
            self._step("place_ack")
            acknowledged = self.place_acknowledged
        elif self.phase == "retreat":
            self._step("retreat_ack")
            acknowledged = self.retreat_acknowledged
        else:  # pragma: no cover - a lifecycle bug would make this reachable
            raise AssertionError(f"unexpected telemetry phase: {self.phase}")
        return SimpleNamespace(
            zero_latched=self.phase != "alignment" and self.zero_latched,
            wheel_stopped=self.phase != "alignment" and self.wheel_stopped,
            stream_running=True,
            target_acknowledged=acknowledged,
            acknowledged_command_sequence=len(self.events),
        )

    def close(self) -> bool:
        self.events.append("teardown:close")
        if self.close_failures_remaining:
            self.close_failures_remaining -= 1
            raise RuntimeError("forced teardown:close failure")
        if self.close_false_remaining:
            self.close_false_remaining -= 1
            return False
        return True


_DESCENT_PLAN = object()


def _runtime(
    *,
    fail_at: str | None = None,
) -> tuple[PlacementLifecycleRuntime, _Controller]:
    controller = _Controller(fail_at=fail_at)

    def release_alignment() -> None:
        controller._step("alignment_release")

    return (
        PlacementLifecycleRuntime(
            controller=controller,
            release_alignment=release_alignment,
        ),
        controller,
    )


def _authorize_release(controller: _Controller, result: bool = True) -> Callable[[], bool]:
    def wait_for_release_authorization() -> bool:
        controller._step("place_release_authorization")
        return result

    return wait_for_release_authorization


def _alignment_fact(runtime: PlacementLifecycleRuntime) -> PlaceAlignmentStoppedAndReleased:
    evidence = runtime.stop_alignment_for_place()
    assert isinstance(evidence, PlaceAlignmentStoppedAndReleased)
    return evidence


def _place_fact(
    runtime: PlacementLifecycleRuntime,
    controller: _Controller,
) -> PlaceAcknowledgedAndReleased:
    evidence = runtime.execute_place(
        _alignment_fact(runtime),
        descent_plan=_DESCENT_PLAN,
        await_release_authorization=_authorize_release(controller),
    )
    assert isinstance(evidence, PlaceAcknowledgedAndReleased)
    return evidence


def _assert_no_manipulation(controller: _Controller) -> None:
    assert "place_command" not in controller.events
    assert "retreat_command" not in controller.events


def test_success_facts_are_phase_specific_runtime_bound_and_ordered() -> None:
    runtime, controller = _runtime()

    with pytest.raises(
        PlacementLifecycleError,
        match="requires PlaceAlignmentStoppedAndReleased evidence",
    ):
        runtime.execute_place(
            None,  # type: ignore[arg-type]
            descent_plan=_DESCENT_PLAN,
            await_release_authorization=_authorize_release(controller),
        )
    _assert_no_manipulation(controller)

    alignment = runtime.stop_alignment_for_place()
    assert alignment.place_alignment_stopped_and_released is True
    assert controller.events == [
        "exact_zero_latch",
        "exact_zero_ack",
        "measured_wheel_stop",
        "alignment_release",
    ]

    # Reading the same completed handoff cannot repeat stop/release side effects.
    assert runtime.stop_alignment_for_place() is alignment
    assert controller.events == [
        "exact_zero_latch",
        "exact_zero_ack",
        "measured_wheel_stop",
        "alignment_release",
    ]

    forged_alignment = replace(alignment)
    assert forged_alignment is not alignment
    with pytest.raises(PlacementLifecycleError, match="not emitted by this runtime"):
        runtime.execute_place(
            forged_alignment,
            descent_plan=_DESCENT_PLAN,
            await_release_authorization=_authorize_release(controller),
        )
    _assert_no_manipulation(controller)

    placed = runtime.execute_place(
        alignment,
        descent_plan=_DESCENT_PLAN,
        await_release_authorization=_authorize_release(controller),
    )
    assert placed.place_acknowledged_and_released is True
    assert controller.events[-3:] == [
        "place_command",
        "place_ack",
        "place_release_authorization",
    ]

    forged_place = replace(placed)
    assert forged_place is not placed
    with pytest.raises(PlacementLifecycleError, match="not emitted by this runtime"):
        runtime.execute_retreat(forged_place)
    assert controller.events.count("retreat_command") == 0

    completed = runtime.execute_retreat(placed)
    assert isinstance(completed, RetreatCompleted)
    assert completed.retreat_completed is True
    assert controller.events[-2:] == ["retreat_command", "retreat_ack"]

    # Terminal calls may return the same fact or reject a duplicate, but must
    # never issue a second posture command.
    before_repeat = list(controller.events)
    try:
        repeated = runtime.execute_retreat(placed)
    except PlacementLifecycleError as exc:
        assert "already" in str(exc)
    else:
        assert repeated is completed
    assert controller.events == before_repeat

    runtime.close()
    after_close = list(controller.events)
    runtime.close()
    assert controller.events == after_close
    assert controller.events.count("teardown:close") == 1


def test_cross_runtime_facts_cannot_authorize_place_or_retreat() -> None:
    source, source_controller = _runtime()
    target, target_controller = _runtime()

    source_alignment = _alignment_fact(source)
    with pytest.raises(PlacementLifecycleError, match="not emitted by this runtime"):
        target.execute_place(
            source_alignment,
            descent_plan=_DESCENT_PLAN,
            await_release_authorization=_authorize_release(target_controller),
        )
    _assert_no_manipulation(target_controller)

    source_place = source.execute_place(
        source_alignment,
        descent_plan=_DESCENT_PLAN,
        await_release_authorization=_authorize_release(source_controller),
    )
    target_alignment = _alignment_fact(target)
    target_place = target.execute_place(
        target_alignment,
        descent_plan=_DESCENT_PLAN,
        await_release_authorization=_authorize_release(target_controller),
    )
    assert target_place is not source_place

    with pytest.raises(PlacementLifecycleError, match="not emitted by this runtime"):
        target.execute_retreat(source_place)
    assert target_controller.events.count("retreat_command") == 0

    source.close()
    target.close()


@pytest.mark.parametrize(
    ("failed_step", "expected_prefix"),
    [
        ("exact_zero_latch", ["exact_zero_latch"]),
        ("exact_zero_ack", ["exact_zero_latch", "exact_zero_ack"]),
        (
            "measured_wheel_stop",
            ["exact_zero_latch", "exact_zero_ack", "measured_wheel_stop"],
        ),
        (
            "alignment_release",
            [
                "exact_zero_latch",
                "exact_zero_ack",
                "measured_wheel_stop",
                "alignment_release",
            ],
        ),
    ],
)
def test_each_alignment_stop_failure_blocks_all_manipulation(
    failed_step: str,
    expected_prefix: list[str],
) -> None:
    runtime, controller = _runtime(fail_at=failed_step)

    with pytest.raises(PlacementLifecycleError, match=failed_step):
        runtime.stop_alignment_for_place()

    assert controller.events == expected_prefix
    with pytest.raises(
        PlacementLifecycleError,
        match="requires PlaceAlignmentStoppedAndReleased evidence",
    ):
        runtime.execute_place(
            None,  # type: ignore[arg-type]
            descent_plan=_DESCENT_PLAN,
            await_release_authorization=_authorize_release(controller),
        )
    _assert_no_manipulation(controller)

    runtime.close()
    after_close = list(controller.events)
    runtime.close()
    assert controller.events == after_close
    assert controller.events.count("teardown:close") == 1


def test_unacknowledged_exact_zero_blocks_wheel_wait_and_release() -> None:
    runtime, controller = _runtime()
    controller.zero_acknowledged = False

    with pytest.raises(PlacementLifecycleError, match="exact-zero.*acknowledged"):
        runtime.stop_alignment_for_place()

    assert controller.events == ["exact_zero_latch", "exact_zero_ack"]
    _assert_no_manipulation(controller)


def test_unlatched_exact_zero_blocks_wheel_wait_and_release() -> None:
    runtime, controller = _runtime()
    controller.zero_latched = False

    with pytest.raises(PlacementLifecycleError, match="exact-zero.*latched"):
        runtime.stop_alignment_for_place()

    assert controller.events == ["exact_zero_latch", "exact_zero_ack"]
    _assert_no_manipulation(controller)


def test_unstopped_wheels_block_alignment_release_and_manipulation() -> None:
    runtime, controller = _runtime()
    controller.wheel_stopped = False

    with pytest.raises(PlacementLifecycleError, match="wheel.*stop"):
        runtime.stop_alignment_for_place()

    assert controller.events == [
        "exact_zero_latch",
        "exact_zero_ack",
        "measured_wheel_stop",
    ]
    assert "alignment_release" not in controller.events
    _assert_no_manipulation(controller)


@pytest.mark.parametrize(
    ("failure", "message", "authorization_calls"),
    [
        ("place_command", "place_command", 0),
        ("place_ack", "place_ack", 0),
        ("place_release_authorization", "place_release_authorization", 1),
    ],
)
def test_place_failure_never_authorizes_retreat_or_retries_the_place_command(
    failure: str,
    message: str,
    authorization_calls: int,
) -> None:
    runtime, controller = _runtime()
    alignment = _alignment_fact(runtime)
    controller.fail_at = failure

    with pytest.raises(PlacementLifecycleError, match=message):
        runtime.execute_place(
            alignment,
            descent_plan=_DESCENT_PLAN,
            await_release_authorization=_authorize_release(controller),
        )

    assert controller.events.count("place_command") == 1
    assert controller.events.count("place_release_authorization") == authorization_calls
    assert controller.events.count("retreat_command") == 0
    before_retry = list(controller.events)

    with pytest.raises(PlacementLifecycleError, match="already"):
        runtime.execute_place(
            alignment,
            descent_plan=_DESCENT_PLAN,
            await_release_authorization=_authorize_release(controller),
        )
    assert controller.events == before_retry
    runtime.close()


def test_false_place_ack_blocks_release_authorization_and_retreat() -> None:
    runtime, controller = _runtime()
    alignment = _alignment_fact(runtime)
    controller.place_acknowledged = False

    with pytest.raises(PlacementLifecycleError, match="place.*acknowledged"):
        runtime.execute_place(
            alignment,
            descent_plan=_DESCENT_PLAN,
            await_release_authorization=_authorize_release(controller),
        )

    assert controller.events.count("place_command") == 1
    assert controller.events.count("place_ack") == 1
    assert controller.events.count("place_release_authorization") == 0
    assert controller.events.count("retreat_command") == 0


def test_false_release_authorization_blocks_retreat() -> None:
    runtime, controller = _runtime()
    alignment = _alignment_fact(runtime)

    with pytest.raises(PlacementLifecycleError, match="release.*authorized"):
        runtime.execute_place(
            alignment,
            descent_plan=_DESCENT_PLAN,
            await_release_authorization=_authorize_release(controller, False),
        )

    assert controller.events.count("place_command") == 1
    assert controller.events.count("place_ack") == 1
    assert controller.events.count("place_release_authorization") == 1
    assert controller.events.count("retreat_command") == 0


@pytest.mark.parametrize(
    ("failure", "acknowledged", "message"),
    [
        ("retreat_command", True, "retreat_command"),
        (None, False, "retreat.*acknowledged"),
    ],
)
def test_retreat_completion_requires_command_and_ack_and_is_never_retried(
    failure: str | None,
    acknowledged: bool,
    message: str,
) -> None:
    runtime, controller = _runtime()
    placed = _place_fact(runtime, controller)
    controller.fail_at = failure
    controller.retreat_acknowledged = acknowledged

    with pytest.raises(PlacementLifecycleError, match=message):
        runtime.execute_retreat(placed)

    assert controller.events.count("retreat_command") == 1
    before_retry = list(controller.events)
    with pytest.raises(PlacementLifecycleError, match="already"):
        runtime.execute_retreat(placed)
    assert controller.events == before_retry
    runtime.close()


@pytest.mark.parametrize("failure_mode", ["exception", "false"])
def test_close_failure_retries_only_until_cleanup_succeeds(failure_mode: str) -> None:
    runtime, controller = _runtime()
    if failure_mode == "exception":
        controller.close_failures_remaining = 1
    else:
        controller.close_false_remaining = 1

    with pytest.raises(PlacementLifecycleError, match="close"):
        runtime.close()
    assert controller.events == ["teardown:close"]

    runtime.close()
    assert controller.events == ["teardown:close", "teardown:close"]

    after_success = list(controller.events)
    runtime.close()
    assert controller.events == after_success


def test_start_calls_prepare_exactly_once_and_is_idempotent() -> None:
    controller = _Controller()
    calls: list[str] = []

    def prepare() -> bool:
        calls.append("prepare")
        return True

    runtime = PlacementLifecycleRuntime(
        controller=controller,
        release_alignment=lambda: None,
        prepare=prepare,
    )
    runtime.start()
    runtime.start()
    assert calls == ["prepare"]
    runtime.close()


@pytest.mark.parametrize("failure_mode", ["exception", "false"])
def test_failed_prepare_is_wrapped_and_never_retried(failure_mode: str) -> None:
    controller = _Controller()
    calls: list[str] = []

    def prepare() -> bool:
        calls.append("prepare")
        if failure_mode == "exception":
            raise RuntimeError("forced prepare failure")
        return False

    runtime = PlacementLifecycleRuntime(
        controller=controller,
        release_alignment=lambda: None,
        prepare=prepare,
    )
    with pytest.raises(PlacementLifecycleError, match="prepare"):
        runtime.start()
    with pytest.raises(PlacementLifecycleError, match="already"):
        runtime.start()
    assert calls == ["prepare"]
    runtime.close()


def test_close_success_prevents_a_later_start() -> None:
    controller = _Controller()
    calls: list[str] = []
    runtime = PlacementLifecycleRuntime(
        controller=controller,
        release_alignment=lambda: None,
        prepare=lambda: calls.append("prepare"),
    )

    runtime.close()
    with pytest.raises(PlacementLifecycleError, match="already closed"):
        runtime.start()
    assert calls == []


@pytest.mark.parametrize("failure_mode", ["exception", "false"])
def test_close_failure_blocks_every_later_lifecycle_operation(
    failure_mode: str,
) -> None:
    runtime, controller = _runtime()
    if failure_mode == "exception":
        controller.close_failures_remaining = 1
    else:
        controller.close_false_remaining = 1

    with pytest.raises(PlacementLifecycleError, match="close"):
        runtime.close()
    assert controller.events == ["teardown:close"]

    release_authorization_calls: list[str] = []

    def authorize_release() -> bool:
        release_authorization_calls.append("authorize")
        return True

    later_operations = (
        runtime.start,
        runtime.stop_alignment_for_place,
        lambda: runtime.execute_place(
            PlaceAlignmentStoppedAndReleased(),
            descent_plan=_DESCENT_PLAN,
            await_release_authorization=authorize_release,
        ),
        lambda: runtime.execute_retreat(PlaceAcknowledgedAndReleased()),
    )
    for operation in later_operations:
        with pytest.raises(PlacementLifecycleError, match="close is pending"):
            operation()

    assert controller.events == ["teardown:close"]
    assert release_authorization_calls == []
    _assert_no_manipulation(controller)
    assert controller.events.count("exact_zero_latch") == 0
    assert controller.events.count("alignment_release") == 0

    # Cleanup itself remains retryable, then becomes idempotent after success.
    runtime.close()
    assert controller.events == ["teardown:close", "teardown:close"]
    after_success = list(controller.events)
    runtime.close()
    assert controller.events == after_success
