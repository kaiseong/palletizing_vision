"""Per-slot configuration: slot 1 is demonstrated, the rest are declared empty."""

from __future__ import annotations

import json
import pathlib

import pytest

from parcel_pose_common.operation_authority import OperationNotAuthorized
from parcel_pose_placing.pallet_control import PalletControlConfig
from parcel_pose_placing.pallet_models import (
    load_slot1_hole_reference,
    require_slot_member,
    slot_config,
)

ROOT_CONFIG_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "configs"
    / "placing_config.json"
)


def root_config() -> dict:
    return json.loads(ROOT_CONFIG_PATH.read_text(encoding="utf-8"))


def test_the_shipped_config_declares_the_four_planned_slots() -> None:
    slots = root_config()["pallet"]["slots"]
    assert sorted(slots, key=int) == ["1", "2", "5", "6"]


def test_slot_one_carries_every_demonstrated_member() -> None:
    root = root_config()
    for member in (
        "hole_reference",
        "offset_right_far_m",
        "long_axis",
        "ready_pose_rad",
        "place_pose_deg",
        "retreat_pose_deg",
    ):
        assert require_slot_member(root, 1, member) is not None


def test_a_slot_without_a_ready_demonstration_is_refused_by_name() -> None:
    """The refusal has to say which slot and which member, before any motion."""

    root = root_config()
    for slot in (2, 6):
        assert slot_config(root, slot) is not None, "the slot is declared"
        with pytest.raises(ValueError, match=f"slot {slot} has no demonstrated"):
            require_slot_member(root, slot, "ready_pose_rad")
    assert require_slot_member(root, 5, "ready_pose_rad") is not None


def test_a_slot_nobody_declared_is_refused_with_the_list() -> None:
    with pytest.raises(ValueError, match="declared slots are 1, 2, 5, 6"):
        slot_config(root_config(), 3)


def test_the_default_slot_supplies_the_postures() -> None:
    root = root_config()
    assert root["pallet"]["default_slot"] == 1
    config = PalletControlConfig.from_root_config(root)
    assert config.place_pose is not None
    assert config.retreat_pose is not None


def test_selecting_an_undemonstrated_slot_yields_no_postures() -> None:
    """Nothing crashes, but placement cannot start without a retreat posture."""

    config = PalletControlConfig.from_root_config(root_config(), slot=5)
    assert config.place_pose is None
    assert config.retreat_pose is None


def test_the_hole_reference_is_read_per_slot() -> None:
    reference = load_slot1_hole_reference(root_config(), 1)
    assert reference.center_base_xy_m == pytest.approx((0.865, 0.139523))
    with pytest.raises(ValueError, match="slot 5 has no demonstrated hole_reference"):
        load_slot1_hole_reference(root_config(), 5)


# --------------------------------------------------------------------------- #
# the operator entry point selects a slot
# --------------------------------------------------------------------------- #
def test_the_live_command_accepts_a_slot() -> None:
    from parcel_pose_placing.pallet_cli import build_parser

    args = build_parser().parse_args(["live", "--headless", "--execute", "--slot", "5"])
    assert args.slot == 5
    assert args.execute is True

    default = build_parser().parse_args(["live"])
    assert default.slot is None, "no slot means pallet.default_slot"
    assert default.execute is False, "perception only by default"


def test_running_an_undemonstrated_slot_is_refused_before_the_sdk_loads() -> None:
    """The refusal must precede every import and print, or it gets buried."""

    import sys

    from box_pallet import place_box

    before = set(sys.modules)
    with pytest.raises(
        OperationNotAuthorized,
        match="slot 5 place live refused; missing fields: "
        "hole_reference, place_pose, retreat_pose",
    ):
        place_box(root_config(), execute=True, ensure_slot1_ready=True, slot=5)
    assert "rby1_sdk" not in set(sys.modules) - before


def test_running_an_undeclared_slot_lists_the_declared_ones() -> None:
    from box_pallet import place_box

    with pytest.raises(ValueError, match="declared slots are 1, 2, 5, 6"):
        place_box(root_config(), slot=3)


def test_the_refusal_tells_you_what_to_put_where() -> None:
    """The error is the discovery mechanism for adding a slot."""

    from parcel_pose_placing.pallet_models import SLOT_MEMBER_SHAPES

    root = root_config()
    members = sorted(root["pallet"]["slots"]["1"])
    assert sorted(SLOT_MEMBER_SHAPES) == members, "every member needs a shape hint"

    for member in members:
        with pytest.raises(ValueError) as caught:
            require_slot_member(root, 2, member)
        message = str(caught.value)
        assert f"pallet.slots.2.{member}" in message, message
        assert SLOT_MEMBER_SHAPES[member] in message, message

    for member in (
        "hole_reference",
        "offset_right_far_m",
        "long_axis",
        "place_pose_deg",
        "retreat_pose_deg",
    ):
        with pytest.raises(ValueError) as caught:
            require_slot_member(root, 5, member)
        assert f"pallet.slots.5.{member}" in str(caught.value)


def test_the_posture_units_are_stated_in_the_hint() -> None:
    """rad versus deg is the mistake this hint exists to prevent."""

    from parcel_pose_placing.pallet_models import SLOT_MEMBER_SHAPES

    assert "RADIANS" in SLOT_MEMBER_SHAPES["ready_pose_rad"]
    assert "DEGREES" in SLOT_MEMBER_SHAPES["place_pose_deg"]
    assert "DEGREES" in SLOT_MEMBER_SHAPES["retreat_pose_deg"]
