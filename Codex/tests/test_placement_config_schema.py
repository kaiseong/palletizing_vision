"""The shipped slot-1 config must build both placement config surfaces."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from parcel_pose.pallet_control import PalletControlConfig
from parcel_pose.pallet_place import PlacementConfig


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "rby1m_v1_2_pallet_slot1_nominal.json"
)


@pytest.fixture(scope="module")
def root_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_defaults_ship_a_descent_free_release() -> None:
    """The library default stays the safest setting: no vertical command."""

    defaults = PlacementConfig()
    assert defaults.maximum_planned_descent_m == 0.0
    assert defaults.maximum_release_gap_m == pytest.approx(0.120)


def test_shipped_config_builds_both_placement_surfaces(root_config: dict) -> None:
    placement = PlacementConfig.from_root_config(root_config)
    control = PalletControlConfig.from_root_config(root_config)
    # The commissioned descent cap and gap limit are site-tuned, so assert the
    # invariants that keep them physically consistent instead of fixed numbers.
    assert placement.maximum_planned_descent_m <= placement.maximum_descent_m
    assert placement.maximum_release_gap_m >= placement.pre_motion_clearance_floor_m
    assert placement.maximum_planned_descent_m < placement.maximum_release_gap_m
    # Slot-1 commissioning seats the carton and stops there: the hands are not
    # opened afterwards, so the spread is zero on both surfaces.
    # Slot-1 withdraws the hands to a demonstrated posture instead of spreading
    # them, so there is no spread distance or opening axis left to configure.
    assert control.retreat_pose is not None
    assert not hasattr(control, "placement_release_spread_m")
    assert not hasattr(placement, "release_spread_m")


def test_unknown_placement_keys_are_rejected(root_config: dict) -> None:
    broken = dict(root_config)
    broken["placement"] = {
        **root_config["placement"],
        "maximum_planed_descent_m": 0.02,
    }
    with pytest.raises(ValueError, match="unknown placement configuration key"):
        PlacementConfig.from_root_config(broken)


def test_new_keys_fall_back_to_defaults(root_config: dict) -> None:
    stripped = {
        key: value
        for key, value in root_config["placement"].items()
        if key
        not in {
            "maximum_planned_descent_m",
            "maximum_release_gap_m",
            "arm_send_once_timeout_s",
        }
    }
    placement = PlacementConfig.from_root_config({**root_config, "placement": stripped})
    control = PalletControlConfig.from_root_config(
        {**root_config, "placement": stripped}
    )
    defaults = PlacementConfig()
    assert control.arm_send_once_timeout_s == PalletControlConfig().arm_send_once_timeout_s
    assert placement.maximum_planned_descent_m == defaults.maximum_planned_descent_m
    assert placement.maximum_release_gap_m == defaults.maximum_release_gap_m


def test_release_gap_below_the_clearance_floor_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be below the clearance floor"):
        PlacementConfig(maximum_release_gap_m=0.040)


def test_planned_descent_cap_cannot_exceed_the_descent_ceiling() -> None:
    with pytest.raises(ValueError, match="cannot exceed maximum_descent_m"):
        PlacementConfig(maximum_planned_descent_m=0.400)
