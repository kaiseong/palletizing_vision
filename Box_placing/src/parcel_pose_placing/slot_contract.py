"""Pure, per-slot data contracts for pallet placement.

This module deliberately performs no robot, camera, controller, or command
construction.  Each selected slot is loaded only from its own JSON block: a
missing posture is represented by ``None`` and is never copied, mirrored, or
synthesized from another slot or from ``robot.ready_pose_rad``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence


PLANNED_SLOTS = (1, 2, 5, 6)
LIVE_REQUIRED_FIELDS = (
    "hole_reference",
    "ready_pose",
    "place_pose",
    "retreat_pose",
)


def _finite_vector(
    value: Sequence[float],
    length: int,
    path: str,
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{path} must contain {length} finite numbers")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must contain {length} finite numbers") from exc
    if len(result) != length or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{path} must contain {length} finite numbers")
    return result


def _pose_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return value


@dataclass(frozen=True, slots=True)
class ReadyPoseContract:
    """A demonstrated ready posture expressed in radians."""

    torso_rad: tuple[float, ...]
    right_arm_rad: tuple[float, ...]
    left_arm_rad: tuple[float, ...]
    head_rad: tuple[float, ...]

    @classmethod
    def from_radians(
        cls,
        value: Mapping[str, Any],
        *,
        path: str,
    ) -> "ReadyPoseContract":
        value = _pose_mapping(value, path)
        return cls(
            torso_rad=_finite_vector(value.get("torso", ()), 6, f"{path}.torso"),
            right_arm_rad=_finite_vector(
                value.get("right_arm", ()), 7, f"{path}.right_arm"
            ),
            left_arm_rad=_finite_vector(
                value.get("left_arm", ()), 7, f"{path}.left_arm"
            ),
            head_rad=_finite_vector(value.get("head", ()), 2, f"{path}.head"),
        )


@dataclass(frozen=True, slots=True)
class PlacementPoseContract:
    """A demonstrated place or retreat posture, normalized to radians."""

    torso_rad: tuple[float, ...]
    right_arm_rad: tuple[float, ...]
    left_arm_rad: tuple[float, ...]

    @classmethod
    def from_degrees(
        cls,
        value: Mapping[str, Any],
        *,
        path: str,
    ) -> "PlacementPoseContract":
        value = _pose_mapping(value, path)

        def radians(name: str, length: int) -> tuple[float, ...]:
            degrees = _finite_vector(value.get(name, ()), length, f"{path}.{name}")
            return tuple(math.radians(item) for item in degrees)

        return cls(
            torso_rad=radians("torso", 6),
            right_arm_rad=radians("right_arm", 7),
            left_arm_rad=radians("left_arm", 7),
        )


@dataclass(frozen=True, slots=True)
class HoleReferenceContract:
    """One independently demonstrated complete-hole reference in the base frame."""

    center_base_xy_m: tuple[float, float]
    yaw_base_rad: float
    axis_branch: str
    reference_frame: str
    source_session: str
    source_selection: str
    source_frame_count: int
    center_std_xy_m: tuple[float, float]
    yaw_std_rad: float
    calibration_status: str


@dataclass(frozen=True, slots=True)
class SlotContract:
    """Independent readiness data for one declared placement slot."""

    slot: int
    hole_reference: HoleReferenceContract | None
    offset_right_far_m: tuple[float, float] | None
    long_axis: str | None
    ready_pose: ReadyPoseContract | None
    place_pose: PlacementPoseContract | None
    retreat_pose: PlacementPoseContract | None

    @property
    def available_fields(self) -> tuple[str, ...]:
        return tuple(
            name for name in LIVE_REQUIRED_FIELDS if getattr(self, name) is not None
        )

    @property
    def missing_fields(self) -> tuple[str, ...]:
        return tuple(
            name for name in LIVE_REQUIRED_FIELDS if getattr(self, name) is None
        )

    @property
    def live_ready(self) -> bool:
        return not self.missing_fields


def _nonempty_text(value: Any, path: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{path} must not be empty")
    return result


def _load_hole_reference(
    raw: Mapping[str, Any],
    *,
    path: str,
    expected_axis_branch: str,
) -> HoleReferenceContract:
    raw = _pose_mapping(raw, path)
    axis_branch = _nonempty_text(raw.get("axis_branch", ""), f"{path}.axis_branch")
    if axis_branch != expected_axis_branch:
        raise ValueError(f"{path}.axis_branch must match pallet.axis_branch")

    if "yaw_base_rad" in raw:
        yaw_base_rad = float(raw["yaw_base_rad"])
    elif "yaw_base_deg" in raw:
        yaw_base_rad = math.radians(float(raw["yaw_base_deg"]))
    else:
        raise ValueError(f"{path} requires yaw_base_rad or yaw_base_deg")
    if "yaw_std_rad" in raw:
        yaw_std_rad = float(raw["yaw_std_rad"])
    elif "yaw_std_deg" in raw:
        yaw_std_rad = math.radians(float(raw["yaw_std_deg"]))
    else:
        raise ValueError(f"{path} requires yaw_std_rad or yaw_std_deg")
    if not math.isfinite(yaw_base_rad):
        raise ValueError(f"{path}.yaw_base must be finite")
    if not math.isfinite(yaw_std_rad) or yaw_std_rad < 0.0:
        raise ValueError(f"{path}.yaw_std must be finite and non-negative")

    center = _finite_vector(raw.get("center_base_xy_m", ()), 2, f"{path}.center_base_xy_m")
    center_std = _finite_vector(raw.get("center_std_xy_m", ()), 2, f"{path}.center_std_xy_m")
    if any(value < 0.0 for value in center_std):
        raise ValueError(f"{path}.center_std_xy_m must be non-negative")
    count = int(raw.get("source_frame_count", 0))
    if isinstance(raw.get("source_frame_count"), bool) or count < 5:
        raise ValueError(f"{path}.source_frame_count must be at least five")

    return HoleReferenceContract(
        center_base_xy_m=(center[0], center[1]),
        yaw_base_rad=(yaw_base_rad + math.pi / 2.0) % math.pi - math.pi / 2.0,
        axis_branch=axis_branch,
        reference_frame=_nonempty_text(
            raw.get("reference_frame", ""), f"{path}.reference_frame"
        ),
        source_session=_nonempty_text(
            raw.get("source_session", ""), f"{path}.source_session"
        ),
        source_selection=_nonempty_text(
            raw.get("source_selection", ""), f"{path}.source_selection"
        ),
        source_frame_count=count,
        center_std_xy_m=(center_std[0], center_std[1]),
        yaw_std_rad=yaw_std_rad,
        calibration_status=_nonempty_text(
            raw.get("calibration_status", ""), f"{path}.calibration_status"
        ),
    )


def _slots_block(root_config: Mapping[str, Any]) -> tuple[Mapping[str, Any], str]:
    if not isinstance(root_config, Mapping):
        raise TypeError("root_config must be a mapping")
    pallet = root_config.get("pallet")
    if not isinstance(pallet, Mapping):
        raise ValueError("pallet configuration block must be an object")
    slots = pallet.get("slots")
    if not isinstance(slots, Mapping):
        raise ValueError("pallet.slots must be an object keyed by slot number")
    axis_branch = _nonempty_text(pallet.get("axis_branch", ""), "pallet.axis_branch")
    return slots, axis_branch


def load_slot_contract(root_config: Mapping[str, Any], slot: int) -> SlotContract:
    """Load only ``pallet.slots.<slot>``; no cross-slot or global fallback."""

    slots, axis_branch = _slots_block(root_config)
    slot_number = int(slot)
    key = str(slot_number)
    if key not in slots:
        declared = ", ".join(sorted((str(item) for item in slots), key=int))
        raise ValueError(
            f"pallet.slots has no slot {key}; declared slots are {declared}"
        )
    raw = slots[key]
    if not isinstance(raw, Mapping):
        raise ValueError(f"pallet.slots.{key} must be an object")
    path = f"pallet.slots.{key}"

    hole_raw = raw.get("hole_reference")
    offset_raw = raw.get("offset_right_far_m")
    long_axis_raw = raw.get("long_axis")
    ready_raw = raw.get("ready_pose_rad")
    place_raw = raw.get("place_pose_deg")
    retreat_raw = raw.get("retreat_pose_deg")
    return SlotContract(
        slot=slot_number,
        hole_reference=(
            None
            if hole_raw is None
            else _load_hole_reference(
                hole_raw,
                path=f"{path}.hole_reference",
                expected_axis_branch=axis_branch,
            )
        ),
        offset_right_far_m=(
            None
            if offset_raw is None
            else _finite_vector(
                offset_raw, 2, f"{path}.offset_right_far_m"
            )
        ),
        long_axis=(
            None
            if long_axis_raw is None
            else _nonempty_text(long_axis_raw, f"{path}.long_axis")
        ),
        ready_pose=(
            None
            if ready_raw is None
            else ReadyPoseContract.from_radians(
                ready_raw, path=f"{path}.ready_pose_rad"
            )
        ),
        place_pose=(
            None
            if place_raw is None
            else PlacementPoseContract.from_degrees(
                place_raw, path=f"{path}.place_pose_deg"
            )
        ),
        retreat_pose=(
            None
            if retreat_raw is None
            else PlacementPoseContract.from_degrees(
                retreat_raw, path=f"{path}.retreat_pose_deg"
            )
        ),
    )


def load_slot_contracts(
    root_config: Mapping[str, Any],
) -> Mapping[int, SlotContract]:
    """Load every declared slot into a read-only mapping of independent objects."""

    slots, _ = _slots_block(root_config)
    contracts = {
        int(key): load_slot_contract(root_config, int(key))
        for key in sorted(slots, key=lambda value: int(value))
    }
    return MappingProxyType(contracts)


__all__ = [
    "HoleReferenceContract",
    "LIVE_REQUIRED_FIELDS",
    "PLANNED_SLOTS",
    "PlacementPoseContract",
    "ReadyPoseContract",
    "SlotContract",
    "load_slot_contract",
    "load_slot_contracts",
]
