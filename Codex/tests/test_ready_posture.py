"""Slot-1 ready posture: the approved posture must be locked in two places."""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np
import pytest

from parcel_pose.pallet_control import (
    READY_LEFT_ARM_RAD,
    READY_RIGHT_ARM_RAD,
    PalletControlConfig,
    ReadyPose,
)

# Negating joints 1, 2, 4 and 6 turns the right arm into the left arm.  Every
# demonstrated slot-1 posture follows this, so a posture that does not is a
# transcription error rather than a new design.
MIRROR = (1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0)

ROOT_CONFIG_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "configs"
    / "rby1m_v1_2_pallet_slot1_nominal.json"
)


def root_config() -> dict:
    return json.loads(ROOT_CONFIG_PATH.read_text(encoding="utf-8"))


def test_the_ready_arms_are_mirror_images() -> None:
    expected = tuple(
        value * sign for value, sign in zip(READY_RIGHT_ARM_RAD, MIRROR, strict=True)
    )
    assert READY_LEFT_ARM_RAD == pytest.approx(expected, abs=1e-12)


def test_the_ready_arms_are_the_demonstrated_degrees() -> None:
    demonstrated_right_deg = (-63.9, -23.8, -3.6, -59.0, -57.1, 63.4, -14.0)
    np.testing.assert_allclose(
        np.degrees(READY_RIGHT_ARM_RAD), demonstrated_right_deg, atol=0.01
    )


def test_the_shipped_config_carries_the_approved_posture() -> None:
    """The constant and the config are two independent copies; both must agree."""

    config = PalletControlConfig.from_root_config(root_config())
    assert config.ready_pose == ReadyPose()
    assert config.ready_pose.right_arm_rad == pytest.approx(READY_RIGHT_ARM_RAD)
    assert config.ready_pose.left_arm_rad == pytest.approx(READY_LEFT_ARM_RAD)


def test_a_config_posture_that_differs_is_refused() -> None:
    """The double lock is what stops an unapproved posture reaching the robot."""

    root = root_config()
    tampered = list(root["robot"]["ready_pose_rad"]["right_arm"])
    tampered[0] += 0.02
    root["robot"]["ready_pose_rad"]["right_arm"] = tampered
    with pytest.raises(ValueError, match="differs from the approved slot-1 pose"):
        PalletControlConfig.from_root_config(root)


def test_the_camera_carrying_joints_are_unchanged() -> None:
    """Vision depends on torso and head only, so replay must stay comparable."""

    config = PalletControlConfig.from_root_config(root_config())
    np.testing.assert_allclose(
        np.degrees(config.ready_pose.torso_rad),
        (0.0, 80.8, -86.7, 36.1, 0.0, 0.0),
        atol=0.05,
    )
    assert config.ready_pose.head_rad[0] == pytest.approx(0.0)
    assert config.ready_pose.head_rad[1] == pytest.approx(0.870, abs=1e-6)


def test_the_ready_transition_still_asks_for_three_seconds() -> None:
    config = PalletControlConfig.from_root_config(root_config())
    assert config.ready_transition_minimum_time_s == pytest.approx(3.0)
    assert math.degrees(config.ready_tolerance_rad) == pytest.approx(1.0)
