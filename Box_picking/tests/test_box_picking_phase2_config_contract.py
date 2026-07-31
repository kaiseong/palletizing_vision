"""Phase-2 selection and high-level picking-config ownership tests."""

from __future__ import annotations

from dataclasses import fields
import math
from types import SimpleNamespace

import pytest

import box_picking
from parcel_pose_common.mobile_servo import ServoConfig
from parcel_pose_picking.auto_grab import AutoGrabConfig
from parcel_pose_picking.cli import build_parser


def test_orientation_cli_defaults_horizontal_and_explicit_vertical_wins() -> None:
    parser = build_parser()

    assert parser.parse_args([]).orientation == "horizontal"
    assert parser.parse_args(["--orientation", "vertical"]).orientation == "vertical"
    assert "--orientation {horizontal,vertical}" in parser.format_help()

    with pytest.raises(SystemExit) as error:
        parser.parse_args(["--orientation", "diagonal"])
    assert error.value.code == 2


def test_entrypoint_declares_the_requested_picking_stage_order() -> None:
    assert box_picking.PICKING_STAGE_ORDER == (
        "preflight",
        "authorize",
        "initialize",
        "ready",
        "acquire",
        "perceive",
        "decide_x_y_yaw",
        "record",
        "loop_exit",
        "stop_release_alignment",
        "grasp_lift",
        "teardown",
    )


def test_horizontal_effective_config_migrates_only_unchanged_high_level_values() -> None:
    args = SimpleNamespace(robot_address="robot.test:50051", robot_power="main")
    baseline = ServoConfig()

    resolved = box_picking._horizontal_auto_grab_config(
        args,
        auto_grab_config_type=AutoGrabConfig,
        servo_config_type=ServoConfig,
    )

    assert resolved.address == "robot.test:50051"
    assert resolved.power == "main"
    assert resolved.servo.target_xy_m == (0.740, 0.0)
    assert math.isclose(resolved.servo.target_long_axis_yaw_rad, math.pi / 2.0)
    assert math.isclose(resolved.servo.arrival_inner_m, 0.010)
    assert math.isclose(resolved.servo.arrival_yaw_inner_rad, math.radians(3.0))

    entrypoint_owned = {
        "target_xy_m",
        "target_long_axis_yaw_rad",
        "arrival_inner_m",
        "arrival_yaw_inner_rad",
    }
    for config_field in fields(ServoConfig):
        if config_field.name not in entrypoint_owned:
            assert getattr(resolved.servo, config_field.name) == getattr(
                baseline,
                config_field.name,
            )


def test_xy_arrival_owner_is_one_radial_tolerance_not_invented_axis_gates() -> None:
    assert box_picking.HORIZONTAL_PICK_ARRIVAL_RADIUS_M == 0.010
    assert not hasattr(box_picking, "HORIZONTAL_PICK_X_TOLERANCE_M")
    assert not hasattr(box_picking, "HORIZONTAL_PICK_Y_TOLERANCE_M")
