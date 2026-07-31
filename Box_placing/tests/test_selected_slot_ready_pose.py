"""Phase-3 contract for independent selected-slot ready postures."""

from __future__ import annotations

import ast
import inspect
import json
import math
import pathlib

import pytest

from parcel_pose_placing import pallet_ready, pallet_runtime
from parcel_pose_placing.pallet_control import (
    PalletControlConfig,
    ReadyPose,
)


ROOT_CONFIG_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "configs"
    / "placing_config.json"
)

SLOT5_READY_POSE_RAD = {
    "torso": (0.000, 1.427, -1.099, -0.214, 0.000, -0.000),
    "right_arm": (-1.124, -0.362, -0.149, -0.906, -0.858, 1.359, -0.038),
    "left_arm": (-1.129, 0.335, 0.160, -0.914, 0.827, 1.359, 0.050),
    "head": (0.000, 0.870),
}


def root_config() -> dict:
    return json.loads(ROOT_CONFIG_PATH.read_text(encoding="utf-8"))


def _pose_members(pose: ReadyPose) -> dict[str, tuple[float, ...]]:
    return {
        "torso": pose.torso_rad,
        "right_arm": pose.right_arm_rad,
        "left_arm": pose.left_arm_rad,
        "head": pose.head_rad,
    }


def test_slot_one_ready_pose_is_consumed_element_by_element() -> None:
    root = root_config()
    expected = root["pallet"]["slots"]["1"]["ready_pose_rad"]

    config = PalletControlConfig.from_root_config(root, slot=1)

    for member, actual in _pose_members(config.ready_pose).items():
        assert actual == tuple(expected[member])


def test_slot_five_ready_pose_is_parsed_exactly_without_motion_construction() -> None:
    config = PalletControlConfig.from_root_config(root_config(), slot=5)

    assert config.selected_slot == 5
    assert _pose_members(config.ready_pose) == SLOT5_READY_POSE_RAD
    assert config.place_pose is None
    assert config.retreat_pose is None


def test_slot_five_left_arm_is_not_mirrored_or_synthesized() -> None:
    pose = PalletControlConfig.from_root_config(root_config(), slot=5).ready_pose
    assert math.copysign(1.0, pose.torso_rad[-1]) == -1.0
    slot1_mirror_signs = (1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0)
    mirrored_right = tuple(
        value * sign
        for value, sign in zip(
            SLOT5_READY_POSE_RAD["right_arm"], slot1_mirror_signs, strict=True
        )
    )

    assert pose.left_arm_rad == SLOT5_READY_POSE_RAD["left_arm"]
    assert pose.left_arm_rad != mirrored_right


@pytest.mark.parametrize("slot", (2, 6))
def test_missing_selected_slot_ready_pose_never_falls_back(slot: int) -> None:
    root = root_config()
    assert "ready_pose_rad" not in root["robot"]
    assert root["pallet"]["slots"]["1"]["ready_pose_rad"] is not None

    with pytest.raises(
        ValueError,
        match=rf"pallet\.slots\.{slot}\.ready_pose_rad must be a mapping",
    ):
        PalletControlConfig.from_root_config(root, slot=slot)


def test_legacy_root_without_slot_contract_has_no_ready_pose_fallback() -> None:
    root = root_config()
    del root["pallet"]["slots"]

    with pytest.raises(
        ValueError,
        match=r"pallet\.slots\.1\.ready_pose_rad must be a mapping",
    ):
        PalletControlConfig.from_root_config(root)


def test_ready_helper_forwards_the_selected_slot_before_any_sdk_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class SelectedSlotObserved(Exception):
        pass

    def stop_after_config_selection(
        root: object,
        *,
        address_override: str | None,
        slot: int | None,
    ) -> PalletControlConfig:
        observed.update(root=root, address=address_override, slot=slot)
        raise SelectedSlotObserved

    monkeypatch.setattr(
        pallet_ready.PalletControlConfig,
        "from_root_config",
        stop_after_config_selection,
    )

    with pytest.raises(SelectedSlotObserved):
        pallet_ready.ensure_slot1_ready_from_config(
            root_config(),
            address="offline:50051",
            power=".*",
            slot=5,
        )

    assert observed["slot"] == 5


def test_runtime_forwards_plan_selected_slot_to_ready_helper() -> None:
    source = inspect.getsource(pallet_runtime.align_and_place)
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ensure_slot1_ready_from_config"
    ]
    assert len(calls) == 1
    slot_keyword = next(
        keyword for keyword in calls[0].keywords if keyword.arg == "slot"
    )
    assert ast.dump(slot_keyword.value, include_attributes=False) == ast.dump(
        ast.Attribute(
            value=ast.Name(id="plan", ctx=ast.Load()),
            attr="selected_slot",
            ctx=ast.Load(),
        ),
        include_attributes=False,
    )
