"""Geometry members remain independently owned by each declared slot."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from parcel_pose_placing.slot_contract import load_slot_contract


CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "placing_config.json"
)


def _root() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_slot1_retains_its_demonstrated_geometry_members() -> None:
    slot1 = load_slot_contract(_root(), 1)
    assert slot1.offset_right_far_m == pytest.approx((0.128, 0.20175))
    assert slot1.long_axis == "u_right"


@pytest.mark.parametrize("slot", (2, 5, 6))
def test_incomplete_slots_do_not_inherit_slot1_geometry(slot: int) -> None:
    root = deepcopy(_root())
    root["pallet"]["default_slot"] = 1
    contract = load_slot_contract(root, slot)
    assert contract.offset_right_far_m is None
    assert contract.long_axis is None
