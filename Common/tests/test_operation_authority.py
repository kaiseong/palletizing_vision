"""Pre-construction authority matrix; all tests are software-only."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pytest

from parcel_pose_common.operation_authority import (
    AuthorityCapability,
    AuthorityCapabilityError,
    AuthorityVerdict,
    ConstructionPermissions,
    OperationBranch,
    OperationMode,
    OperationNotAuthorized,
    OperationRequest,
    PickOrientation,
    ReadinessManifest,
    authorize_operation,
    current_readiness_manifest,
)


LIVE_CAPABILITIES = (
    AuthorityCapability.ROBOT_CONNECTION,
    AuthorityCapability.CONTROLLER,
    AuthorityCapability.READY_POSTURE,
    AuthorityCapability.STREAM,
    AuthorityCapability.SEQUENCER,
    AuthorityCapability.DEPENDENT_COMMAND,
)


@dataclass
class CallRecorder:
    calls: list[str] = field(default_factory=list)

    def factory(self, name: str) -> Callable[[], object]:
        def construct() -> object:
            self.calls.append(name)
            return object()

        return construct


def _invoke_after_capability_check(
    verdict: AuthorityVerdict,
    capability: AuthorityCapability,
    factory: Callable[[], object],
) -> object:
    verdict.require_capability(capability)
    return factory()


def _initialize_live_stack(request: OperationRequest, recorder: CallRecorder) -> None:
    """Tiny consumer proving the required verdict-before-factory call order."""

    verdict = authorize_operation(request).require_authorized()
    for capability in LIVE_CAPABILITIES:
        _invoke_after_capability_check(
            verdict,
            capability,
            recorder.factory(capability.value),
        )


@pytest.mark.parametrize(
    ("operation_request", "purpose"),
    [
        (
            OperationRequest.pick(PickOrientation.HORIZONTAL),
            "demonstrated_horizontal_pick",
        ),
        (OperationRequest.place(1), "demonstrated_slot_1_place"),
    ],
)
def test_only_demonstrated_live_branches_are_authorized(
    operation_request: OperationRequest,
    purpose: str,
) -> None:
    verdict = authorize_operation(operation_request)

    assert verdict.allowed is True
    assert verdict.purpose == purpose
    assert verdict.missing_fields == ()
    assert verdict.permissions == ConstructionPermissions(
        robot_connection=True,
        controller=True,
        ready_posture=True,
        stream=True,
        sequencer=True,
        dependent_command=True,
    )
    assert verdict.permissions.any_live_side_effect is True
    assert verdict.require_authorized() is verdict


@pytest.mark.parametrize(
    ("operation_request", "missing_fields", "reason"),
    [
        (
            OperationRequest.pick(PickOrientation.VERTICAL),
            (
                "perception_validation",
                "ready_pose",
                "grasp_pose",
            ),
            "vertical pick live refused; missing fields: "
            "perception_validation, ready_pose, grasp_pose",
        ),
        (
            OperationRequest.place(2),
            ("hole_reference", "ready_pose", "place_pose", "retreat_pose"),
            "slot 2 place live refused; missing fields: "
            "hole_reference, ready_pose, place_pose, retreat_pose",
        ),
        (
            OperationRequest.place(5),
            ("hole_reference", "place_pose", "retreat_pose"),
            "slot 5 place live refused; missing fields: "
            "hole_reference, place_pose, retreat_pose",
        ),
        (
            OperationRequest.place(6),
            ("hole_reference", "ready_pose", "place_pose", "retreat_pose"),
            "slot 6 place live refused; missing fields: "
            "hole_reference, ready_pose, place_pose, retreat_pose",
        ),
    ],
)
def test_incomplete_live_branches_return_stable_exact_missing_fields(
    operation_request: OperationRequest,
    missing_fields: tuple[str, ...],
    reason: str,
) -> None:
    verdict = authorize_operation(operation_request)

    assert verdict.allowed is False
    assert verdict.missing_fields == missing_fields
    assert verdict.reason == reason
    assert verdict.readiness_provenance
    assert verdict.permissions == ConstructionPermissions()
    assert verdict.permissions.any_live_side_effect is False


@pytest.mark.parametrize(
    "operation_request",
    [
        OperationRequest.pick(PickOrientation.VERTICAL),
        OperationRequest.place(2),
        OperationRequest.place(5),
        OperationRequest.place(6),
    ],
)
def test_refusal_happens_before_every_prohibited_factory_or_command(
    operation_request: OperationRequest,
) -> None:
    recorder = CallRecorder()

    with pytest.raises(OperationNotAuthorized) as error:
        _initialize_live_stack(operation_request, recorder)

    assert error.value.verdict.request == operation_request
    assert recorder.calls == []


@pytest.mark.parametrize("mode", [OperationMode.REPLAY, OperationMode.DRY_RUN])
def test_slot_5_offline_modes_grant_only_perception(mode: OperationMode) -> None:
    request = OperationRequest.place(5, mode=mode)
    verdict = authorize_operation(request).require_authorized()
    recorder = CallRecorder()

    assert verdict.purpose == "offline_perception_only"
    assert verdict.missing_fields == ()
    assert verdict.permissions.offline_perception is True
    assert verdict.permissions.any_live_side_effect is False
    _invoke_after_capability_check(
        verdict,
        AuthorityCapability.OFFLINE_PERCEPTION,
        recorder.factory("offline_perception"),
    )

    for capability in LIVE_CAPABILITIES:
        with pytest.raises(AuthorityCapabilityError) as error:
            _invoke_after_capability_check(
                verdict,
                capability,
                recorder.factory(capability.value),
            )
        assert error.value.capability is capability

    assert recorder.calls == ["offline_perception"]


def test_custom_manifest_reports_only_fields_that_are_still_missing() -> None:
    manifest = ReadinessManifest(
        branch=OperationBranch.SLOT_2_PLACE,
        available_fields=frozenset({"ready_pose", "hole_reference"}),
        provenance="unit_test_partial_slot_2",
    )

    verdict = authorize_operation(OperationRequest.place(2), readiness=manifest)

    assert verdict.available_fields == ("hole_reference", "ready_pose")
    assert verdict.missing_fields == ("place_pose", "retreat_pose")
    assert verdict.reason == (
        "slot 2 place live refused; missing fields: place_pose, retreat_pose"
    )
    assert verdict.readiness_provenance == "unit_test_partial_slot_2"


def test_manifest_must_belong_to_the_requested_branch() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "readiness branch mismatch: request=slot_2_place, "
            "manifest=slot_1_place"
        ),
    ):
        authorize_operation(
            OperationRequest.place(2),
            readiness=current_readiness_manifest(OperationBranch.SLOT_1_PLACE),
        )


def test_unapproved_mode_fails_closed_with_no_capabilities() -> None:
    verdict = authorize_operation(
        OperationRequest.pick(PickOrientation.HORIZONTAL, mode=OperationMode.REPLAY)
    )

    assert verdict.allowed is False
    assert verdict.missing_fields == ("approved_operation_mode",)
    assert verdict.reason == (
        "horizontal pick replay refused; missing fields: approved_operation_mode"
    )
    assert verdict.permissions == ConstructionPermissions()


@pytest.mark.parametrize("slot", [0, 3, 4, 7])
def test_only_in_scope_slots_can_form_an_operation_request(slot: int) -> None:
    with pytest.raises(ValueError, match="slot must be one of 1, 2, 5, 6"):
        OperationRequest.place(slot)
