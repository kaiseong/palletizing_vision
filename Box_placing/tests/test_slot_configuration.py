"""Per-slot configuration: slot 1 is demonstrated, the rest are declared empty."""

from __future__ import annotations

import json
import pathlib

import pytest

from parcel_pose_placing.pallet_control import PalletControlConfig
from parcel_pose_placing.pallet_models import (
    load_slot1_hole_reference,
    require_slot_member,
    slot_config,
)

ROOT_CONFIG_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "configs"
    / "rby1m_v1_2_pallet_slot1_nominal.json"
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


def test_an_undemonstrated_slot_is_refused_by_name() -> None:
    """The refusal has to say which slot and which member, before any motion."""

    root = root_config()
    for slot in (2, 5, 6):
        assert slot_config(root, slot) is not None, "the slot is declared"
        with pytest.raises(ValueError, match=f"slot {slot} has no demonstrated"):
            require_slot_member(root, slot, "ready_pose_rad")


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

    from parcel_pose_placing.pallet_runtime import run_pallet_live

    before = set(sys.modules)
    with pytest.raises(ValueError, match="slot 5 has no demonstrated hole_reference"):
        run_pallet_live(root_config(), execute=True, ensure_slot1_ready=True, slot=5)
    assert "rby1_sdk" not in set(sys.modules) - before


def test_running_an_undeclared_slot_lists_the_declared_ones() -> None:
    from parcel_pose_placing.pallet_runtime import run_pallet_live

    with pytest.raises(ValueError, match="declared slots are 1, 2, 5, 6"):
        run_pallet_live(root_config(), slot=3)
