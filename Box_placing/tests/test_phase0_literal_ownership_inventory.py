"""Historical guard for the G001 Phase-0 literal ownership inventory."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from parcel_pose_common.mobile_servo import ServoConfig
from parcel_pose_picking.auto_grab import AutoGrabConfig
from parcel_pose_picking.cli import build_parser as build_picking_parser
from parcel_pose_placing.pallet_cli import build_parser as build_placing_parser
from parcel_pose_placing.pallet_models import load_slot1_hole_reference
from parcel_pose_placing.pallet_runtime import resolve_live_plan
from parcel_pose_placing.pallet_servo import PalletServoConfig


REPO_ROOT = Path(__file__).resolve().parents[2]
INVENTORY_PATH = (
    REPO_ROOT / "docs" / "inventory" / "palletizing_literal_ownership_phase0.json"
)
PICKING_CONFIG_PATH = REPO_ROOT / "Box_picking" / "configs" / "picking_config.json"
PLACING_CONFIG_PATH = REPO_ROOT / "Box_placing" / "configs" / "placing_config.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _inventory() -> dict:
    return _load_json(INVENTORY_PATH)


def _rows() -> dict[str, dict]:
    rows = _inventory()["values"]
    assert len({row["id"] for row in rows}) == len(rows)
    return {row["id"]: row for row in rows}


def _current_snapshots() -> dict:
    pick = AutoGrabConfig()
    placing_root = _load_json(PLACING_CONFIG_PATH)
    place = PalletServoConfig.from_root_config(placing_root)
    slot1 = load_slot1_hole_reference(placing_root, 1)
    shared_place_tolerance = {
        "x_m": None,
        "y_m": None,
        "yaw_rad": place.arrival_yaw_inner_rad,
    }
    unavailable_slot = lambda slot: {
        "selection": slot,
        "status": "refused_explicit_null_hole_reference",
        "target": {"x_m": None, "y_m": None, "yaw_rad": None},
        "high_level_tolerance": shared_place_tolerance,
        "current_gate": {"constructed": False},
    }
    return {
        "pick": {
            "horizontal": {
                "selection": "horizontal",
                "status": "current_implicit_entrypoint_default",
                "target": {
                    "x_m": pick.servo.target_xy_m[0],
                    "y_m": pick.servo.target_xy_m[1],
                    "yaw_rad": pick.servo.target_long_axis_yaw_rad,
                },
                "high_level_tolerance": {
                    "x_m": None,
                    "y_m": None,
                    "yaw_rad": pick.servo.arrival_yaw_inner_rad,
                },
                "current_gate": {
                    "xy_metric": "euclidean_norm",
                    "xy_inner_m": pick.servo.arrival_inner_m,
                    "xy_outer_m": pick.servo.arrival_outer_m,
                    "yaw_inner_rad": pick.servo.arrival_yaw_inner_rad,
                    "yaw_outer_rad": pick.servo.arrival_yaw_outer_rad,
                    "grasp_residual_deg": pick.max_grasp_yaw_residual_deg,
                },
            },
            "vertical": {
                "selection": None,
                "status": "missing_entrypoint_branch_and_cli_selector",
                "target": {"x_m": None, "y_m": None, "yaw_rad": None},
                "high_level_tolerance": {
                    "x_m": None,
                    "y_m": None,
                    "yaw_rad": None,
                },
                "current_gate": None,
            },
        },
        "place": {
            "1": {
                "selection": 1,
                "status": "current_default_resolves",
                "target": {
                    "x_m": slot1.center_base_xy_m[0],
                    "y_m": slot1.center_base_xy_m[1],
                    "yaw_rad": slot1.yaw_base_rad,
                },
                "high_level_tolerance": shared_place_tolerance,
                "current_gate": {
                    "xy_metric": "euclidean_norm",
                    "xy_inner_m": place.arrival_inner_m,
                    "xy_outer_m": place.arrival_outer_m,
                    "yaw_inner_rad": place.arrival_yaw_inner_rad,
                    "yaw_outer_rad": place.arrival_yaw_outer_rad,
                },
            },
            "2": unavailable_slot(2),
            "5": unavailable_slot(5),
            "6": unavailable_slot(6),
        },
    }


def test_literal_rows_match_current_sources() -> None:
    inventory = _inventory()
    rows = _rows()
    pick = ServoConfig()
    placing = _load_json(PLACING_CONFIG_PATH)
    place_servo = PalletServoConfig.from_root_config(placing)
    slot1 = placing["pallet"]["slots"]["1"]["hole_reference"]

    expected = {
        "selection.pick_orientation": "horizontal",
        "selection.placing_slot": placing["pallet"]["default_slot"],
        "pick.target.x_m": pick.target_xy_m[0],
        "pick.target.y_m": pick.target_xy_m[1],
        "pick.target.yaw_rad": pick.target_long_axis_yaw_rad,
        "pick.tolerance.x_m": None,
        "pick.tolerance.y_m": None,
        "pick.tolerance.yaw_rad": pick.arrival_yaw_inner_rad,
        "place.slot1.target.x_m": slot1["center_base_xy_m"][0],
        "place.slot1.target.y_m": slot1["center_base_xy_m"][1],
        "place.slot1.target.yaw_deg": slot1["yaw_base_deg"],
        "place.tolerance.x_m": None,
        "place.tolerance.y_m": None,
        "place.tolerance.yaw_rad": place_servo.arrival_yaw_inner_rad,
    }
    for slot in (2, 5, 6):
        assert placing["pallet"]["slots"][str(slot)]["hole_reference"] is None
        for component in ("x_m", "y_m", "yaw_deg"):
            expected[f"place.slot{slot}.target.{component}"] = None

    assert set(rows) == set(expected)
    assert {row_id: row["value"] for row_id, row in rows.items()} == expected
    for row in inventory["values"]:
        assert row["fallback"]
        assert row["action_planned"]
        assert row["test"]
        owner_path = row["current_owner"].split("#", 1)[0].split(" + ", 1)[0]
        assert (REPO_ROOT / owner_path).is_file(), row["id"]


def test_snapshots_are_deterministic_and_match_phase0_defaults() -> None:
    first = _current_snapshots()
    second = _current_snapshots()
    assert first == second == _inventory()["branch_snapshots"]
    assert json.dumps(first, sort_keys=True, separators=(",", ":")) == json.dumps(
        second, sort_keys=True, separators=(",", ":")
    )


def test_missing_axis_and_branch_values_are_explicit_nulls() -> None:
    inventory = _inventory()
    rows = _rows()
    placing = _load_json(PLACING_CONFIG_PATH)

    for row_id in (
        "pick.tolerance.x_m",
        "pick.tolerance.y_m",
        "place.tolerance.x_m",
        "place.tolerance.y_m",
    ):
        assert rows[row_id]["value"] is None
        assert rows[row_id]["source_literal"] is None
    vertical = inventory["branch_snapshots"]["pick"]["vertical"]
    assert vertical["selection"] is None
    assert set(vertical["target"].values()) == {None}
    for slot in (2, 5, 6):
        assert placing["pallet"]["slots"][str(slot)]["hole_reference"] is None
        snapshot = inventory["branch_snapshots"]["place"][str(slot)]
        assert set(snapshot["target"].values()) == {None}
        assert snapshot["current_gate"] == {"constructed": False}


def test_current_cli_and_loader_precedence_is_recorded_truthfully(
    tmp_path: Path,
) -> None:
    inventory = _inventory()
    cli_rows = {row["id"]: row for row in inventory["cli_options"]}
    assert set(cli_rows) == {
        "cli.pick.config",
        "cli.place.live.config",
        "cli.place.live.slot",
    }

    alternate = tmp_path / "alternate.json"
    pick_parser = build_picking_parser()
    assert pick_parser.parse_args([]).config == PICKING_CONFIG_PATH
    assert pick_parser.parse_args(["--config", str(alternate)]).config == alternate
    assert pick_parser.parse_args([]).orientation == "horizontal"
    assert pick_parser.parse_args(["--orientation", "vertical"]).orientation == "vertical"
    assert "--orientation" in pick_parser.format_help()
    phase0_orientation = _rows()["selection.pick_orientation"]
    assert "No CLI/config selection exists" in phase0_orientation["fallback"]
    assert "Later move the explicit default/selection" in phase0_orientation["action_planned"]
    assert cli_rows["cli.pick.config"]["affects"] == []

    place_parser = build_placing_parser()
    defaults = place_parser.parse_args(["live"])
    explicit = place_parser.parse_args(
        ["live", "--config", str(alternate), "--slot", "1", "--headless"]
    )
    assert defaults.config == PLACING_CONFIG_PATH
    assert defaults.slot is None
    assert explicit.config == alternate
    assert explicit.slot == 1

    placing = _load_json(PLACING_CONFIG_PATH)
    config_default_is_missing = deepcopy(placing)
    config_default_is_missing["pallet"]["default_slot"] = 2
    plan = resolve_live_plan(
        auto_place_slot1=False,
        controller=None,
        ensure_slot1_ready=False,
        execute=False,
        headless=True,
        log_jsonl=None,
        max_frames=1,
        output_mp4=None,
        root_config=config_default_is_missing,
        slot=1,
        warmup_frames=0,
    )
    assert plan.selected_slot == 1, "explicit slot currently wins over JSON default"
    with pytest.raises(ValueError, match="slot 2 has no demonstrated hole_reference"):
        resolve_live_plan(
            auto_place_slot1=False,
            controller=None,
            ensure_slot1_ready=False,
            execute=False,
            headless=True,
            log_jsonl=None,
            max_frames=1,
            output_mp4=None,
            root_config=config_default_is_missing,
            slot=None,
            warmup_frames=0,
        )

    both_yaw_units = deepcopy(placing)
    both_yaw_units["pallet"]["slots"]["1"]["hole_reference"]["yaw_base_rad"] = 0.25
    assert load_slot1_hole_reference(both_yaw_units, 1).yaw_base_rad == 0.25
    assert both_yaw_units["pallet"]["slots"]["1"]["hole_reference"]["yaw_base_deg"] == -90.0

    without_json_tolerance = deepcopy(placing)
    del without_json_tolerance["servo"]["arrival_yaw_inner_rad"]
    assert PalletServoConfig.from_root_config(
        without_json_tolerance
    ).arrival_yaw_inner_rad == math.radians(3.0)
    assert inventory["current_precedence"]["place_slot_selection"] == [
        "explicit CLI/programmatic slot",
        "selected JSON pallet.default_slot",
        "resolve_live_plan code fallback 1",
    ]
