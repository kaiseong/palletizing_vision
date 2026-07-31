"""Phase-3 software-only contract tests for independent placement slots."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from parcel_pose_placing.slot_contract import (
    LIVE_REQUIRED_FIELDS,
    PLANNED_SLOTS,
    load_slot_contract,
    load_slot_contracts,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "placing_config.json"
)

SLOT5_READY = {
    "torso_rad": (0.0, 1.427, -1.099, -0.214, 0.0, -0.0),
    "right_arm_rad": (-1.124, -0.362, -0.149, -0.906, -0.858, 1.359, -0.038),
    "left_arm_rad": (-1.129, 0.335, 0.160, -0.914, 0.827, 1.359, 0.050),
    "head_rad": (0.0, 0.870),
}


def _root() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_shipped_slot_contracts_are_independent_and_read_only() -> None:
    contracts = load_slot_contracts(_root())
    assert tuple(contracts) == PLANNED_SLOTS
    assert [contracts[slot].slot for slot in PLANNED_SLOTS] == list(PLANNED_SLOTS)
    assert len({id(contracts[slot]) for slot in PLANNED_SLOTS}) == len(PLANNED_SLOTS)
    with pytest.raises(TypeError):
        contracts[3] = contracts[1]  # type: ignore[index]


def test_slot1_contract_remains_complete_with_its_exact_shipped_data() -> None:
    root = _root()
    contract = load_slot_contract(root, 1)
    raw = root["pallet"]["slots"]["1"]

    assert contract.live_ready is True
    assert contract.available_fields == LIVE_REQUIRED_FIELDS
    assert contract.missing_fields == ()
    assert contract.hole_reference is not None
    assert contract.hole_reference.center_base_xy_m == pytest.approx((0.865, 0.139523))
    assert contract.hole_reference.yaw_base_rad == pytest.approx(-math.pi / 2.0)
    assert contract.ready_pose is not None
    assert contract.ready_pose.torso_rad == tuple(raw["ready_pose_rad"]["torso"])
    assert contract.ready_pose.right_arm_rad == tuple(
        raw["ready_pose_rad"]["right_arm"]
    )
    assert contract.ready_pose.left_arm_rad == tuple(
        raw["ready_pose_rad"]["left_arm"]
    )
    assert contract.ready_pose.head_rad == tuple(raw["ready_pose_rad"]["head"])
    assert contract.place_pose is not None
    assert contract.retreat_pose is not None


def test_slot5_retains_the_operator_supplied_ready_pose_elementwise() -> None:
    contract = load_slot_contract(_root(), 5)
    assert contract.ready_pose is not None
    for field, expected in SLOT5_READY.items():
        assert getattr(contract.ready_pose, field) == expected
    assert math.copysign(1.0, contract.ready_pose.torso_rad[-1]) == -1.0

    assert contract.hole_reference is None
    assert contract.place_pose is None
    assert contract.retreat_pose is None
    assert contract.available_fields == ("ready_pose",)
    assert contract.missing_fields == (
        "hole_reference",
        "place_pose",
        "retreat_pose",
    )
    assert contract.live_ready is False


@pytest.mark.parametrize("slot", (2, 6))
def test_slots_without_demonstrations_keep_every_live_field_unavailable(slot: int) -> None:
    contract = load_slot_contract(_root(), slot)
    assert contract.hole_reference is None
    assert contract.ready_pose is None
    assert contract.place_pose is None
    assert contract.retreat_pose is None
    assert contract.available_fields == ()
    assert contract.missing_fields == LIVE_REQUIRED_FIELDS
    assert contract.live_ready is False


def test_missing_slot_values_never_fall_back_to_global_or_another_slot() -> None:
    root = deepcopy(_root())
    root["robot"]["ready_pose_rad"] = deepcopy(
        root["pallet"]["slots"]["1"]["ready_pose_rad"]
    )
    root["pallet"]["default_slot"] = 1

    slot2 = load_slot_contract(root, 2)
    assert slot2.ready_pose is None
    assert slot2.hole_reference is None
    assert slot2.place_pose is None
    assert slot2.retreat_pose is None


def test_partial_ready_pose_is_rejected_instead_of_mirrored_or_synthesized() -> None:
    root = deepcopy(_root())
    del root["pallet"]["slots"]["5"]["ready_pose_rad"]["left_arm"]

    with pytest.raises(
        ValueError,
        match=r"pallet\.slots\.5\.ready_pose_rad\.left_arm must contain 7",
    ):
        load_slot_contract(root, 5)


def test_slot5_ready_does_not_activate_missing_place_or_retreat() -> None:
    root = _root()
    raw = root["pallet"]["slots"]["5"]
    assert raw["hole_reference"] is None
    assert raw["place_pose_deg"] is None
    assert raw["retreat_pose_deg"] is None
    assert load_slot_contract(root, 5).live_ready is False
