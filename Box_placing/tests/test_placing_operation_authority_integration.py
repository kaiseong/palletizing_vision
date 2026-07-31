"""Software-only authority gates at the public placing entrypoint.

The tests deliberately replace the complete live runtime with fakes.  A refused
branch must stop at the pure authority decision, before even the first runtime
stage can construct acquisition, controller, ready, stream, or sequencer state.
Slot-5 replay/dry-run is kept as a pure capability contract here; the Phase-5
offline replay runner is intentionally not invented by this Phase-3 test.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Callable
import sys

import pytest

import box_pallet
from parcel_pose_common.operation_authority import (
    AuthorityCapability,
    AuthorityCapabilityError,
    ConstructionPermissions,
    OperationMode,
    OperationNotAuthorized,
)


_LIVE_CAPABILITIES = (
    AuthorityCapability.ROBOT_CONNECTION,
    AuthorityCapability.CONTROLLER,
    AuthorityCapability.READY_POSTURE,
    AuthorityCapability.STREAM,
    AuthorityCapability.SEQUENCER,
    AuthorityCapability.DEPENDENT_COMMAND,
)


@dataclass
class _Effects:
    authority_calls: list[tuple[int, OperationMode]] = field(default_factory=list)
    runtime_stages: list[str] = field(default_factory=list)
    constructors: Counter[str] = field(default_factory=Counter)
    commands: Counter[str] = field(default_factory=Counter)


def _runtime_trap(effects: _Effects) -> ModuleType:
    """Return the legacy runtime surface with every stage observable and fatal."""

    module = ModuleType("parcel_pose_placing.pallet_runtime")

    def forbidden_stage(
        name: str,
        *,
        constructors: tuple[str, ...] = (),
        command: str | None = None,
    ) -> Callable[..., Any]:
        def call(*_args: Any, **_kwargs: Any) -> Any:
            effects.runtime_stages.append(name)
            effects.constructors.update(constructors)
            if command is not None:
                effects.commands[command] += 1
            raise AssertionError(
                f"{name} ran before the incomplete slot was refused"
            )

        return call

    # If authority is accidentally delayed, these model every prohibited class
    # named by the PRD.  The stage-call assertion below additionally catches a
    # partial initialization that happens before one of these constructors.
    module.resolve_live_plan = forbidden_stage(
        "resolve_live_plan",
        constructors=("acquisition",),
    )
    module.assemble_live_stack = forbidden_stage(
        "assemble_live_stack",
        constructors=("controller", "sequencer"),
    )
    module.initial_run_state = forbidden_stage(
        "initial_run_state",
        constructors=("ready_posture",),
    )
    module.align_and_place = forbidden_stage(
        "align_and_place",
        constructors=("robot_connection", "stream"),
        command="place",
    )
    return module


def _trace_public_authority(
    monkeypatch: pytest.MonkeyPatch,
    effects: _Effects,
) -> None:
    real_authorize = box_pallet.authorize_slot_operation

    def traced(slot: int, mode: OperationMode | str):
        normalized_mode = OperationMode(mode)
        effects.authority_calls.append((int(slot), normalized_mode))
        return real_authorize(slot, normalized_mode)

    monkeypatch.setattr(box_pallet, "authorize_slot_operation", traced)


@pytest.mark.parametrize(
    ("slot", "reason"),
    [
        (
            2,
            "slot 2 place live refused; missing fields: "
            "hole_reference, ready_pose, place_pose, retreat_pose",
        ),
        (
            5,
            "slot 5 place live refused; missing fields: "
            "hole_reference, place_pose, retreat_pose",
        ),
        (
            6,
            "slot 6 place live refused; missing fields: "
            "hole_reference, ready_pose, place_pose, retreat_pose",
        ),
    ],
)
def test_incomplete_live_slot_refuses_before_all_runtime_effects(
    monkeypatch: pytest.MonkeyPatch,
    slot: int,
    reason: str,
) -> None:
    effects = _Effects()
    _trace_public_authority(monkeypatch, effects)
    monkeypatch.setitem(
        sys.modules,
        "parcel_pose_placing.pallet_runtime",
        _runtime_trap(effects),
    )

    with pytest.raises(OperationNotAuthorized) as caught:
        box_pallet.place_box(
            {},
            execute=True,
            auto_place_slot1=True,
            ensure_slot1_ready=True,
            slot=slot,
            headless=True,
        )

    assert str(caught.value) == reason
    assert caught.value.verdict.reason == reason
    assert effects.authority_calls == [(slot, OperationMode.LIVE)]
    assert effects.runtime_stages == []
    assert effects.constructors == Counter()
    assert effects.commands == Counter()


def _invoke_capability(
    verdict: Any,
    capability: AuthorityCapability,
    action: Callable[[], None],
) -> None:
    verdict.require_capability(capability)
    action()


@pytest.mark.parametrize("mode", [OperationMode.REPLAY, OperationMode.DRY_RUN])
def test_slot5_offline_authority_allows_only_perception_setup(
    mode: OperationMode,
) -> None:
    verdict = box_pallet.authorize_slot_operation(5, mode)
    constructors: Counter[str] = Counter()
    commands: Counter[str] = Counter()

    assert verdict.allowed is True
    assert verdict.reason == (
        f"slot 5 place {mode.value} authorized for offline_perception_only"
    )
    assert verdict.permissions == ConstructionPermissions(offline_perception=True)

    _invoke_capability(
        verdict,
        AuthorityCapability.OFFLINE_PERCEPTION,
        lambda: constructors.update(("offline_perception",)),
    )
    for capability in _LIVE_CAPABILITIES:
        counter = commands if capability is AuthorityCapability.DEPENDENT_COMMAND else constructors
        name = "place" if capability is AuthorityCapability.DEPENDENT_COMMAND else capability.value
        with pytest.raises(AuthorityCapabilityError) as caught:
            _invoke_capability(
                verdict,
                capability,
                lambda counter=counter, name=name: counter.update((name,)),
            )
        assert caught.value.capability is capability

    assert constructors == Counter({"offline_perception": 1})
    for prohibited in (
        "robot_connection",
        "controller",
        "ready_posture",
        "stream",
        "sequencer",
    ):
        assert constructors[prohibited] == 0
    assert commands["place"] == 0
