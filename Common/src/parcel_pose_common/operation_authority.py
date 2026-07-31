"""Pure pre-construction authority for palletizing demo branches.

This module intentionally depends only on the Python standard library.  A caller
must obtain an authorized verdict before constructing robot, controller, ready
posture, stream, or sequencer objects.  Offline slot-5 replay is a separate
authority: it permits perception work, never robot construction or commands.

The manifests below describe the first-pass data state approved by the project
plan.  Missing poses or references are never mirrored, synthesized, or borrowed
from another branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Iterable, Mapping

def _field_names(values: Iterable[str], owner: str) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError(f"{owner} cannot contain an empty field name")
        result.append(normalized)
    if len(result) != len(set(result)):
        raise ValueError(f"{owner} cannot contain duplicate field names")
    return tuple(result)


class OperationMode(str, Enum):
    """Execution context used when authority is decided."""

    LIVE = "live"
    REPLAY = "replay"
    DRY_RUN = "dry_run"


class PickOrientation(str, Enum):
    """Supported first-pass box orientation selections."""

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


class OperationBranch(str, Enum):
    """Branches explicitly covered by the approved demo authority matrix."""

    HORIZONTAL_PICK = "horizontal_pick"
    VERTICAL_PICK = "vertical_pick"
    SLOT_1_PLACE = "slot_1_place"
    SLOT_2_PLACE = "slot_2_place"
    SLOT_5_PLACE = "slot_5_place"
    SLOT_6_PLACE = "slot_6_place"

    @property
    def label(self) -> str:
        return _BRANCH_LABELS[self]


class AuthorityCapability(str, Enum):
    """Construction or action classes controlled by an authority verdict."""

    ROBOT_CONNECTION = "robot_connection"
    CONTROLLER = "controller"
    READY_POSTURE = "ready_posture"
    STREAM = "stream"
    SEQUENCER = "sequencer"
    DEPENDENT_COMMAND = "dependent_command"
    OFFLINE_PERCEPTION = "offline_perception"


@dataclass(frozen=True, slots=True)
class OperationRequest:
    """One branch and mode, resolved before any side-effectful construction."""

    branch: OperationBranch
    mode: OperationMode

    def __post_init__(self) -> None:
        object.__setattr__(self, "branch", OperationBranch(self.branch))
        object.__setattr__(self, "mode", OperationMode(self.mode))

    @classmethod
    def pick(
        cls,
        orientation: PickOrientation | str,
        *,
        mode: OperationMode | str = OperationMode.LIVE,
    ) -> "OperationRequest":
        orientation = PickOrientation(orientation)
        branch = {
            PickOrientation.HORIZONTAL: OperationBranch.HORIZONTAL_PICK,
            PickOrientation.VERTICAL: OperationBranch.VERTICAL_PICK,
        }[orientation]
        return cls(branch=branch, mode=OperationMode(mode))

    @classmethod
    def place(
        cls,
        slot: int,
        *,
        mode: OperationMode | str = OperationMode.LIVE,
    ) -> "OperationRequest":
        try:
            normalized_slot = int(slot)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"slot must be one of 1, 2, 5, 6; got {slot!r}") from exc
        branches = {
            1: OperationBranch.SLOT_1_PLACE,
            2: OperationBranch.SLOT_2_PLACE,
            5: OperationBranch.SLOT_5_PLACE,
            6: OperationBranch.SLOT_6_PLACE,
        }
        try:
            branch = branches[normalized_slot]
        except KeyError as exc:
            raise ValueError(
                f"slot must be one of 1, 2, 5, 6; got {normalized_slot}"
            ) from exc
        return cls(branch=branch, mode=OperationMode(mode))


@dataclass(frozen=True, slots=True)
class ReadinessManifest:
    """Auditable data fields currently present for one independent branch."""

    branch: OperationBranch
    available_fields: frozenset[str]
    provenance: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "branch", OperationBranch(self.branch))
        normalized = frozenset(_field_names(self.available_fields, "available_fields"))
        object.__setattr__(self, "available_fields", normalized)
        provenance = str(self.provenance).strip()
        if not provenance:
            raise ValueError("readiness provenance must be non-empty")
        object.__setattr__(self, "provenance", provenance)


@dataclass(frozen=True, slots=True)
class ConstructionPermissions:
    """What may happen after the verdict; all defaults deliberately deny."""

    robot_connection: bool = False
    controller: bool = False
    ready_posture: bool = False
    stream: bool = False
    sequencer: bool = False
    dependent_command: bool = False
    offline_perception: bool = False

    @property
    def any_live_side_effect(self) -> bool:
        return any(
            (
                self.robot_connection,
                self.controller,
                self.ready_posture,
                self.stream,
                self.sequencer,
                self.dependent_command,
            )
        )

    def permits(self, capability: AuthorityCapability | str) -> bool:
        capability = AuthorityCapability(capability)
        return bool(getattr(self, capability.value))


@dataclass(frozen=True, slots=True)
class AuthorityVerdict:
    """Stable result of readiness and mode authorization."""

    request: OperationRequest
    allowed: bool
    purpose: str
    required_fields: tuple[str, ...]
    available_fields: tuple[str, ...]
    missing_fields: tuple[str, ...]
    readiness_provenance: str
    reason: str
    permissions: ConstructionPermissions

    def __post_init__(self) -> None:
        required = tuple(_field_names(self.required_fields, "required_fields"))
        available = tuple(_field_names(self.available_fields, "available_fields"))
        missing = tuple(_field_names(self.missing_fields, "missing_fields"))
        object.__setattr__(self, "required_fields", required)
        object.__setattr__(self, "available_fields", available)
        object.__setattr__(self, "missing_fields", missing)
        if not self.reason.strip():
            raise ValueError("authority reason must be non-empty")
        if self.allowed and missing:
            raise ValueError("an authorized verdict cannot have missing fields")
        if not self.allowed and self.permissions != ConstructionPermissions():
            raise ValueError("a refused verdict cannot grant any capability")
        if self.request.mode is not OperationMode.LIVE and self.permissions.any_live_side_effect:
            raise ValueError("offline modes cannot grant live side effects")

    def require_authorized(self) -> "AuthorityVerdict":
        """Return this verdict or fail before the caller constructs anything."""

        if not self.allowed:
            raise OperationNotAuthorized(self)
        return self

    def require_capability(
        self, capability: AuthorityCapability | str
    ) -> "AuthorityVerdict":
        """Require one granted capability before invoking its factory/action."""

        capability = AuthorityCapability(capability)
        self.require_authorized()
        if not self.permissions.permits(capability):
            raise AuthorityCapabilityError(self, capability)
        return self


class OperationNotAuthorized(RuntimeError):
    """Raised by :meth:`AuthorityVerdict.require_authorized` on refusal."""

    def __init__(self, verdict: AuthorityVerdict) -> None:
        self.verdict = verdict
        super().__init__(verdict.reason)


class AuthorityCapabilityError(RuntimeError):
    """Raised when an authorized mode does not grant a requested capability."""

    def __init__(
        self,
        verdict: AuthorityVerdict,
        capability: AuthorityCapability,
    ) -> None:
        self.verdict = verdict
        self.capability = capability
        super().__init__(
            f"{verdict.request.branch.label} {verdict.request.mode.value} "
            f"does not grant {capability.value}"
        )


@dataclass(frozen=True, slots=True)
class _AuthorityPolicy:
    purpose: str
    required_fields: tuple[str, ...]
    permissions: ConstructionPermissions


_BRANCH_LABELS: Mapping[OperationBranch, str] = MappingProxyType(
    {
        OperationBranch.HORIZONTAL_PICK: "horizontal pick",
        OperationBranch.VERTICAL_PICK: "vertical pick",
        OperationBranch.SLOT_1_PLACE: "slot 1 place",
        OperationBranch.SLOT_2_PLACE: "slot 2 place",
        OperationBranch.SLOT_5_PLACE: "slot 5 place",
        OperationBranch.SLOT_6_PLACE: "slot 6 place",
    }
)

_LIVE_PERMISSIONS = ConstructionPermissions(
    robot_connection=True,
    controller=True,
    ready_posture=True,
    stream=True,
    sequencer=True,
    dependent_command=True,
)
_OFFLINE_PERCEPTION_PERMISSIONS = ConstructionPermissions(offline_perception=True)

_PICK_FIELDS = (
    "perception_validation",
    "ready_pose",
    "grasp_pose",
)
_PLACE_FIELDS = ("hole_reference", "ready_pose", "place_pose", "retreat_pose")
_SLOT_5_REPLAY_FIELDS = (
    "rgbd_recordings",
    "camera_intrinsics",
    "camera_extrinsics",
)

_POLICIES: Mapping[tuple[OperationBranch, OperationMode], _AuthorityPolicy] = (
    MappingProxyType(
        {
            (OperationBranch.HORIZONTAL_PICK, OperationMode.LIVE): _AuthorityPolicy(
                purpose="demonstrated_horizontal_pick",
                required_fields=_PICK_FIELDS,
                permissions=_LIVE_PERMISSIONS,
            ),
            (OperationBranch.VERTICAL_PICK, OperationMode.LIVE): _AuthorityPolicy(
                purpose="vertical_pick",
                required_fields=_PICK_FIELDS,
                permissions=_LIVE_PERMISSIONS,
            ),
            (OperationBranch.SLOT_1_PLACE, OperationMode.LIVE): _AuthorityPolicy(
                purpose="demonstrated_slot_1_place",
                required_fields=_PLACE_FIELDS,
                permissions=_LIVE_PERMISSIONS,
            ),
            (OperationBranch.SLOT_2_PLACE, OperationMode.LIVE): _AuthorityPolicy(
                purpose="slot_2_place",
                required_fields=_PLACE_FIELDS,
                permissions=_LIVE_PERMISSIONS,
            ),
            (OperationBranch.SLOT_5_PLACE, OperationMode.LIVE): _AuthorityPolicy(
                purpose="slot_5_place",
                required_fields=_PLACE_FIELDS,
                permissions=_LIVE_PERMISSIONS,
            ),
            (OperationBranch.SLOT_5_PLACE, OperationMode.REPLAY): _AuthorityPolicy(
                purpose="offline_perception_only",
                required_fields=_SLOT_5_REPLAY_FIELDS,
                permissions=_OFFLINE_PERCEPTION_PERMISSIONS,
            ),
            (OperationBranch.SLOT_5_PLACE, OperationMode.DRY_RUN): _AuthorityPolicy(
                purpose="offline_perception_only",
                required_fields=_SLOT_5_REPLAY_FIELDS,
                permissions=_OFFLINE_PERCEPTION_PERMISSIONS,
            ),
            (OperationBranch.SLOT_6_PLACE, OperationMode.LIVE): _AuthorityPolicy(
                purpose="slot_6_place",
                required_fields=_PLACE_FIELDS,
                permissions=_LIVE_PERMISSIONS,
            ),
        }
    )
)

_CURRENT_READINESS: Mapping[OperationBranch, ReadinessManifest] = MappingProxyType(
    {
        OperationBranch.HORIZONTAL_PICK: ReadinessManifest(
            branch=OperationBranch.HORIZONTAL_PICK,
            available_fields=frozenset(_PICK_FIELDS),
            provenance="phase0_existing_horizontal_pick",
        ),
        OperationBranch.VERTICAL_PICK: ReadinessManifest(
            branch=OperationBranch.VERTICAL_PICK,
            available_fields=frozenset({"orientation_estimator"}),
            provenance="phase0_vertical_incomplete",
        ),
        OperationBranch.SLOT_1_PLACE: ReadinessManifest(
            branch=OperationBranch.SLOT_1_PLACE,
            available_fields=frozenset(_PLACE_FIELDS),
            provenance="phase0_existing_slot_1_place",
        ),
        OperationBranch.SLOT_2_PLACE: ReadinessManifest(
            branch=OperationBranch.SLOT_2_PLACE,
            available_fields=frozenset(),
            provenance="phase0_slot_2_explicit_nulls",
        ),
        OperationBranch.SLOT_5_PLACE: ReadinessManifest(
            branch=OperationBranch.SLOT_5_PLACE,
            available_fields=frozenset(
                {
                    "ready_pose",
                    "rgbd_recordings",
                    "camera_intrinsics",
                    "camera_extrinsics",
                }
            ),
            provenance="approved_slot_5_ready_and_recordings",
        ),
        OperationBranch.SLOT_6_PLACE: ReadinessManifest(
            branch=OperationBranch.SLOT_6_PLACE,
            available_fields=frozenset(),
            provenance="phase0_slot_6_explicit_nulls",
        ),
    }
)



def current_readiness_manifest(
    branch: OperationBranch | str,
) -> ReadinessManifest:
    """Return the immutable, first-pass readiness state for ``branch``."""

    return _CURRENT_READINESS[OperationBranch(branch)]


def authorize_operation(
    request: OperationRequest,
    *,
    readiness: ReadinessManifest | None = None,
) -> AuthorityVerdict:
    """Decide authority without importing an SDK or constructing any resource."""

    if not isinstance(request, OperationRequest):
        raise TypeError("request must be an OperationRequest")
    manifest = readiness or current_readiness_manifest(request.branch)
    if manifest.branch is not request.branch:
        raise ValueError(
            "readiness branch mismatch: "
            f"request={request.branch.value}, manifest={manifest.branch.value}"
        )

    policy = _POLICIES.get((request.branch, request.mode))
    if policy is None:
        required_fields = ("approved_operation_mode",)
        missing_fields = required_fields
        purpose = "unsupported"
    else:
        required_fields = policy.required_fields
        missing_fields = tuple(
            field for field in required_fields if field not in manifest.available_fields
        )
        purpose = policy.purpose

    allowed = policy is not None and not missing_fields
    if allowed:
        reason = (
            f"{request.branch.label} {request.mode.value} authorized for {purpose}"
        )
        permissions = policy.permissions
    else:
        reason = (
            f"{request.branch.label} {request.mode.value} refused; missing fields: "
            + ", ".join(missing_fields)
        )
        permissions = ConstructionPermissions()

    return AuthorityVerdict(
        request=request,
        allowed=allowed,
        purpose=purpose,
        required_fields=required_fields,
        available_fields=tuple(sorted(manifest.available_fields)),
        missing_fields=missing_fields,
        readiness_provenance=manifest.provenance,
        reason=reason,
        permissions=permissions,
    )
